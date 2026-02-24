import datetime
from functools import partial
import imp
import os
import json
import numpy as np

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb

from datamil.utils.jax_utils import initialize_compilation_cache
from datamil.data.dataset import make_interleaved_dataset
from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import SaveCallback
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_flags.DEFINE_config_file(
    "config",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Dataset: {FLAGS.config.dataset_kwargs_list}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    pretrained_model = OctoModel.load_pretrained(
        FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step,
    )
    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    config = config.to_dict()
    check_config_diff(config, pretrained_model.config)

    # del config["model"]["observation_tokenizers"]["wrist"]
    if FLAGS.config.get("use_proprio", False):
        from octo.model.components.tokenizers import LowdimObsTokenizer
        config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer,
            n_bins=256,
            bin_type="normal",
            low=-2.0,
            high=2.0,
            obs_keys=["proprio"],
        )
    
    action_chunks = FLAGS.config.get("action_chunks", 4)
    if action_chunks != 4:
        print(f"Changing action head to predict next {action_chunks} actions")
        config["model"]["heads"]["action"]['kwargs'].update(pred_horizon=action_chunks)

    #########
    #
    # Setup Data Loader
    #
    #########

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    # load standardize_fn from `path/to/file.py:fn_name` format
    # if (
    #     standardize_fn := FLAGS.config["dataset_kwargs"].get("standardize_fn", None)
    # ) is not None:
    #     path, name = standardize_fn.split(":")
    #     # imp is deprecated, but it's also what ml_collections uses
    #     standardize_fn = getattr(imp.load_source("standardize_fn", path), name)
    #     del FLAGS.config["dataset_kwargs"]["standardize_fn"]
    #     FLAGS.config["dataset_kwargs"]["standardize_fn"] = standardize_fn

    for i, dataset_kwargs in enumerate(FLAGS.config.dataset_kwargs_list):
        if (standardize_fn := dataset_kwargs.get("standardize_fn", None)) is not None:
            if isinstance(standardize_fn, str):
                path, name = standardize_fn.split(":")
                # imp is deprecated, but it's also what ml_collections uses
                standardize_fn = getattr(imp.load_source("standardize_fn", path), name)
                del dataset_kwargs["standardize_fn"]
                dataset_kwargs["standardize_fn"] = standardize_fn

                FLAGS.config.dataset_kwargs_list[i] = dataset_kwargs
        
        if (training_idxs_path := dataset_kwargs.get("training_idxs_path", None)) is not None:
            assert os.path.exists(training_idxs_path), f"training_idxs_path {training_idxs_path} does not exist"

            # if json file then do this else...
            if training_idxs_path.endswith('.json'):
                with open(training_idxs_path, 'r') as f:
                    training_indices = json.load(f)
                indices = []
                for shard in training_indices['shards']:
                    shard_idx = int(shard['raw_data']['basename'].split('/')[-2].split('_')[-1])
                    indices.append(shard_idx)
            elif training_idxs_path.endswith('.npy'):
                indices = np.load(training_idxs_path).tolist()
            else:
                raise NotImplementedError("Only .json and .npy training_idxs_path are supported currently")

            dataset_kwargs['training_idxs'] = indices
            del dataset_kwargs['training_idxs_path']
        else:
            dataset_kwargs['training_idxs'] = None
        
        FLAGS.config.dataset_kwargs_list[i] = dataset_kwargs

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(FLAGS.config.seed)

    train_data = make_interleaved_dataset(
        FLAGS.config.dataset_kwargs_list,
        sample_weights=FLAGS.config.sample_weights,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
        shuffle_buffer_size=FLAGS.config.shuffle_buffer_size,
        batch_size=FLAGS.config.batch_size,
        balance_weights=False
    )

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    train_data_iter = map(
        shard,
        map(
            process_batch,
            train_data.iterator(prefetch=tf.data.AUTOTUNE),
        ),
    )

    example_batch = next(train_data_iter)
    print(example_batch.keys())

    #########
    #
    # Load Pretrained Model
    #
    #########

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=train_data.dataset_statistics[0],
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    #########
    #
    # Setup Optimizer and Train State
    #
    #########
    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )

    #########
    #
    # Save all metadata
    #
    #########

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"]["pad_mask"].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        with open(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train, # set it to false to turn off dropout (check if this is correct)
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train, # set it to false to turn off dropout (check if this is correct)
        )
        return action_loss, action_metrics


    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        # Gradient Metrics (TODO: Does the finetuner need these?) ###
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        # End Debug Metrics #

        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
