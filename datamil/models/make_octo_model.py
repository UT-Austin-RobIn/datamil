import imp
import os
import types
import json
from functools import cache
from absl import app, flags
import flax
import jax
from fnmatch import fnmatch
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
import numpy as np

from octo.model.octo_model import OctoModel
from octo.model.components.action_heads import DiscreteActionHead, L1ActionHead
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    check_config_diff,
    merge_params,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

# patch the diffusion action head with per_sample_loss function
from datamil.models.octo_action_head import diffusion_per_sample_loss
from octo.model.components.action_heads import DiffusionActionHead
DiffusionActionHead.per_sample_loss = diffusion_per_sample_loss

import datamil.metagradients.flags_config as flags_config
FLAGS = flags.FLAGS

from typing import Callable

# @cache
def make_model(data_batcher: Callable):

    # initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    ############################################
    ############ get text processor ############
    ############################################
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

    for batch in data_batcher(0, 1, None):
        example_batch = batch.batch
        break

    if FLAGS.config.get('use_proprio', False):
        from octo.model.components.tokenizers import LowdimObsTokenizer
        config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer,
            n_bins=256,
            bin_type="normal",
            low=-2.0,
            high=2.0,
            obs_keys=["proprio"],
        )
    
    action_chunks = FLAGS.config.get('action_chunks', 4)
    if action_chunks != 4:
        print(f"Changing action head to predict next {action_chunks} actions")
        config["model"]["heads"]["action"]['kwargs'].update(pred_horizon=action_chunks)
    
    loss_type = FLAGS.config.get('loss_type', 'mse')
    if loss_type == "l1":
        print("Changing action head to predict using L1 loss")
        config["model"]["heads"]["action"] = ModuleSpec.create(
            L1ActionHead,
            readout_key="readout_action",
            pred_horizon=action_chunks,
            action_dim=7
        )
    elif loss_type == "discrete":
        print("Changing action head to predict using discrete loss")
        config["model"]["heads"]["action"] = ModuleSpec.create(
            DiscreteActionHead,
            readout_key="readout_action",
            use_map=False,
            token_per="",
            pred_horizon=1,
            action_dim=7,
            vocab_size=48,
            normalization_type="normal",
        )

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()
    ############################################

    ds_stat_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
        'dataset_statistics.json'
    )

    with open(ds_stat_path, 'r') as f:
       dataset_statistics = json.load(f)

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
        dataset_statistics=dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    frozen_keys = FLAGS.config.optimizer.frozen_keys

    flat_params = flax.traverse_util.flatten_dict(params)
    frozen_params = {}
    trainable_params = {}
    for key in flat_params:
        flat_key = ".".join(key)

        match = False
        for frozen_key in frozen_keys:
            if fnmatch(flat_key, frozen_key):
                match = True

        if match:
            frozen_params[key] = flat_params[key]
        else:
            trainable_params[key] = flat_params[key]

    frozen_params = flax.traverse_util.unflatten_dict(frozen_params)
    trainable_params = flax.traverse_util.unflatten_dict(trainable_params)

    return model, frozen_params, trainable_params