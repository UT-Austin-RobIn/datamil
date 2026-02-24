import os
import jax
import flax
import numpy as np
from copy import deepcopy
import tensorflow as tf

from functools import partial
from functools import cache
import json

from octo.utils.jax_utils import initialize_compilation_cache
from datamil.metagradients.make_libero_loader_mds import make_split_loader_and_data_weights, make_replay_dataset
from datamil.models.make_octo_model import make_model
from datamil.metagradients.core.vjp_robodm import vjp_robodm
from datamil.metagradients.core.optimizers.adam import make_adam_optimizer
from datamil.metagradients.core.optimizers.interpolation import interp_from, interp_from_mom
from datamil.metagradients.core.utils import make_shardings
from datamil.metagradients.core.vjp_blocks import example_loss_vjp_skeleton, sample_loss_vjp_head

from absl import flags, app
from ml_collections import config_flags

# jax.config.update("jax_disable_jit", True)

import datamil.metagradients.flags_config as flags_config
FLAGS = flags.FLAGS

EPS = 1.0000000000000001e-11
SEED_SPACING = 100000

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def grad_from_store(deps, batch_indices):

    flat_deps = {k: v for d in deps.values() for k, v in d.items()}
    # num_datapoints = max(b.max() for b in batch_indices) + 1
    num_datapoints = max(b.max() for b in batch_indices.values()) + 1
    gradient = np.zeros((num_datapoints,), dtype=np.float32)

    # for i, bixs in enumerate(batch_indices):
        # gradient[bixs] += flat_deps[i]
    for batch_n, indices in batch_indices.items():
        gradient[indices] += flat_deps[batch_n]

    return gradient

@cache
def make_vjp_skele(bs):
    return jax.tree_util.Partial(partial(example_loss_vjp_skeleton, bs=bs))

@partial(jax.jit, static_argnames=['train', 'divisor'])
def per_sample_loss_fn(params,
                       batch,
                       model,
                       frozen_params,
                       train,
                       data_weights=None,
                       divisor=1.0):

    """
    inputs:
        params: trainable parameters
        frozen_params: non-trainable parameters
    """
    assert divisor == 1.0, divisor

    flat_params = flax.traverse_util.flatten_dict(params)
    flat_frozen_params = flax.traverse_util.flatten_dict(frozen_params)

    all_params = flat_params | flat_frozen_params
    all_params = flax.traverse_util.unflatten_dict(all_params)

    model = model.replace(params=all_params)

    _, (data, _) = batch[:2]
    assert 'seed' in data

    seed = data['seed'][0]
    rng = jax.random.PRNGKey(seed)

    bound_module = model.module.bind({"params": all_params}, rngs={"dropout": rng})
    transformer_embeddings = bound_module.octo_transformer(
        data["observation"],
        data["task"],
        data["observation"]["pad_mask"],
        # train=train,
        train=False,
    )
    action_loss, action_metrics = bound_module.heads["action"].per_sample_loss(
        transformer_embeddings,  # Action head knows to pull out the action readout_key
        data["action"],
        pad_mask=data["observation"]["pad_mask"],
        # train=train,
        train=False,
    )

    # if 'weights' in data:
    #     weights = data['weights']
    #     # weights = 2*(weights-1) + 1
    #     # jax.debug.print("B: {}", weights)
    #     action_loss = action_loss * weights
    
    if data_weights is not None:
        indices = data['index']
        these_data_weights = data_weights[indices]
        action_loss = action_loss * these_data_weights

    return action_loss / divisor

def data_selection_iter(data_weights: jax.numpy.array,
                        checkpoint_path: str,
                        job_id: int=0):

    initialize_compilation_cache()

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    print('Checkpoint path:', checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    FLAGS.config.checkpoint_path = checkpoint_path

    train_batcher = partial(make_split_loader_and_data_weights, mode='train', seed=FLAGS.config.seed, iter_seed=job_id)
    train_its = FLAGS.config.num_steps
    bob_its = FLAGS.config.bob_steps
    forward_its = train_its - bob_its

    val_batcher = partial(make_split_loader_and_data_weights, mode='val', seed=FLAGS.config.seed)
    val_its = len(make_replay_dataset(0, None, None, train=False)[0]) 

    model, frozen_params, trainable_params = make_model(train_batcher)

    num_trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(trainable_params))
    num_frozen_params = sum(x.size for x in jax.tree_util.tree_leaves(frozen_params))
    num_total_params = num_trainable_params + num_frozen_params

    print(f'Trainable params: {num_trainable_params:,}')
    print(f'Frozen params: {num_frozen_params:,}')
    print(f'Total params: {num_total_params:,}')

    psl = jax.tree_util.Partial(
        per_sample_loss_fn,
        frozen_params=frozen_params,
        model=model,
    )

    optimizer_dict = FLAGS.config.optimizer.to_dict()
    lr_scheduler_dict = optimizer_dict['learning_rate']

    OPTIMIZER_KWARGS = {
        'lr': lr_scheduler_dict['peak_value'],
        # 'wd': 1e-5,
        'wd': optimizer_dict['weight_decay'],
        'pct_start': lr_scheduler_dict['warmup_steps'] / lr_scheduler_dict['decay_steps'],
        'pct_final': 1,
        'b1': 0.9,
        'b2': 0.95,
        'min_lr_relative': max(lr_scheduler_dict['init_value'], EPS),
        'final_min_lr_relative': max(lr_scheduler_dict['end_value'], EPS),
        'eps': EPS,
        'eps_sqrt': EPS,
        'selective_wd': True,
        'dtype': jax.numpy.float32,
        'factored_lr_wd': False,
        'anneal_type': 'linear',
        'eps_schedule': jax.tree_util.Partial(interp_from, steps=200,
                                              eps0=1e-08, eps_root0=1e-08, space='geometric'),
        'mom_schedule': jax.tree_util.Partial(interp_from_mom, steps=25, mom0=0.85,
                                        mom1=1, space='linear'),
        'per_param_lr': None,
        'reuse_optimizer': False,
    }

    state0 = make_adam_optimizer(
        initial_params=trainable_params,
        train_its=train_its,
        **OPTIMIZER_KWARGS,
    )

    aux_datasets = {}
    return_kw = False

    sharding, replicated_sharding = make_shardings()
    head_val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)

    # vjp_skele = jax.tree_util.Partial(partial(example_loss_vjp_skeleton, bs=FLAGS.config.batch_size))
    vjp_skele = make_vjp_skele
    vjp_head = partial(
        sample_loss_vjp_head,
        per_sample_loss=psl,
        val_batcher=head_val_batcher,
        val_its = val_its,
    )

    vjp_kw = dict(
        state=state0,
        vjp_head=vjp_head,
        vjp_skele=vjp_skele,
        data_weights=data_weights,
        return_kw=return_kw,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        psl=psl,
        n_train_ba=forward_its, # not train_its
        n_val_ba=val_its,
        aux_datasets=aux_datasets,
        return_state=True,
        forward_only=True,
        segment_size=25,
        # forward_only=False,
    )
    import time
    s1 = time.time()
    ret = vjp_robodm(**vjp_kw)
    print('\n', '*'*20)
    print('>> Done with forward pass', time.time()-s1)
    print('*'*20, '\n')

    vjp_kw.update(dict(
        state=ret['final_state'],
        n_train_ba=train_its,
        forward_only=False,
    ))

    s2 = time.time()
    final_ret = vjp_robodm(**vjp_kw)
    print('\n', '*'*20)
    print('>> Done with backward pass', time.time()-s2)
    print('*'*20, '\n')

    y0 = float(final_ret['primal'])
    deps = final_ret['deps']
    batch_indices = final_ret['batch_indices']
    final_state = final_ret['final_state']
    num_datapoints = data_weights.size

    import json
    with open(os.path.join(checkpoint_path, 'hparams_config.json'), 'w') as f:
        json.dump(FLAGS.config.to_dict(), f, indent=4)

    grad = grad_from_store(deps, batch_indices)
    print(grad)
    if len(grad) > num_datapoints:
        print('>> Shrinking grad array from', len(grad), 'to', num_datapoints)
        assert (grad[num_datapoints:] == 0).all()
        grad = grad[:num_datapoints]
        print(grad)
    elif len(grad) < num_datapoints:
        print('>> Growing grad array from', len(grad), 'to', num_datapoints)
        grad = np.concatenate([grad, np.zeros((num_datapoints - len(grad),))])
        print(grad)

    grad_path = os.path.join(checkpoint_path, 'datamodels.npy')
    np.save(grad_path, grad)

    return grad

def create_include_index(candidate_grad):
    index_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
        'index.json'
    )

    previous_include_index_path = FLAGS.config.include_index_path
    with open(previous_include_index_path, 'r') as f:
        previous_include_index = json.load(f)

    with open(index_path, 'r') as f:
        index = json.load(f)

    selected_shards_idxs = set()
    for shard in previous_include_index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[0].split('_')[-1])
        if candidate_grad[shard_idx] <= 0:
            selected_shards_idxs.add(shard_idx)

    for shard in index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[0].split('_')[-1])
        if candidate_grad[shard_idx] < 0: # select the sample
            selected_shards_idxs.add(shard_idx)

    selected_shards = []
    for shard in index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[0].split('_')[-1])
        if shard_idx in selected_shards_idxs:
            selected_shards.append(shard)

    include_index = {
        'shards': selected_shards,
        'version': 2
    }

    print(len(include_index['shards']))

    checkpoint_path = FLAGS.config.checkpoint_path
    include_index_path = os.path.join(
        checkpoint_path,
        'include_index.json'
    )

    with open(include_index_path, 'w') as f:
        json.dump(include_index, f)

def create_include_index_perc():

    if '-' in FLAGS.config.dataset_kwargs.name:
        formatted_date_time = FLAGS.config.folder_name
        all_index_path = os.path.join(
            FLAGS.config.save_dir, 
            FLAGS.config.dataset_kwargs.name, 
            formatted_date_time,
            'merged_index.json'
        )
    else:
        all_index_path = os.path.join(
            FLAGS.config.dataset_kwargs.data_dir,
            FLAGS.config.dataset_kwargs.name,
            'index.json'
        )

    with open(all_index_path, 'r') as f:
        all_index = json.load(f)

    job_id = FLAGS.config.job_id
    include_idx = set([int(x['raw_data']['basename'].split('/')[-2].split('_')[-1]) for x in all_index['shards']])
    for i in range(job_id+1):
        grad_path = os.path.join(
            os.path.dirname(FLAGS.config.checkpoint_path),
            f'iter_{i}',
            'datamodels.npy'
        )

        grad_i = np.load(grad_path)[1_000_000:]
        tmp_grad = grad_i[np.where(grad_i != 0)[0]]

        perc_10 = np.percentile(tmp_grad, 20)
        perc_90 = np.percentile(tmp_grad, 80)
        for shard in all_index['shards']:
            shard_idx = int(shard['raw_data']['basename'].split('/')[-2].split('_')[-1])

            if grad_i[shard_idx] <= perc_10:
                include_idx.add(shard_idx)
            elif grad_i[shard_idx] >= perc_90:
                if shard_idx in include_idx:
                    include_idx.remove(shard_idx)
            else:
                pass

    include_shards = []
    for shard in all_index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[-2].split('_')[-1])
        if shard_idx in include_idx:
            include_shards.append(shard)

    print('Total shards:', len(all_index['shards']))
    print('Included shards:', len(include_shards))

    current_include_index = {
        'shards': include_shards,
        'version': 2
    }

    current_include_index_path = os.path.join(FLAGS.config.checkpoint_path, 'include_index.json')
    with open(current_include_index_path, 'w') as f:
        json.dump(current_include_index, f)


def main(_):

    job_id = FLAGS.config.job_id
    meta_checkpoint_path = os.path.join(FLAGS.config.save_dir, FLAGS.config.dataset_kwargs.name, FLAGS.config.folder_name)

    checkpoint_path = os.path.join(meta_checkpoint_path, f'iter_{job_id}')
    os.makedirs(checkpoint_path, exist_ok=True)

    if job_id == 0:
        _, data_weights = make_replay_dataset(0, 1e5, None, train=True, return_dw_only=True)
        data_weights = jax.numpy.concatenate(
            [data_weights, jax.numpy.zeros_like(data_weights)],
            axis=0
        )

        FLAGS.config.include_index_path =  os.path.join(
            FLAGS.config.dataset_kwargs.data_dir,
            FLAGS.config.dataset_kwargs.name,
            'index.json'
        )

    else:
        prev_job = job_id - 1
        _, data_weights = make_replay_dataset(0, 1e5, None, train=True, return_dw_only=True)
        data_weights = jax.numpy.concatenate(
            [data_weights, jax.numpy.zeros_like(data_weights)],
            axis=0
        )

        prev_checkpoint_path = os.path.join(meta_checkpoint_path, f'iter_{prev_job}')
        assert os.path.exists(prev_checkpoint_path)
        FLAGS.config.include_index_path = os.path.join(
            prev_checkpoint_path,
            'include_index.json'
        )

    grad = data_selection_iter(
        data_weights,
        checkpoint_path,
        job_id=job_id,
    )

    # candidate_grad = grad[1_000_000:]
    # create_include_index(candidate_grad)
    create_include_index_perc()
    
if __name__ == '__main__':
    app.run(main)