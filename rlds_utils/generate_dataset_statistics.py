import os
import numpy as np
import imp
import inspect
import types
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import dlimp as dl

from datamil.data.dataset import restructure, get_dataset_statistics
from datamil import libero_dataset_kwargs as dataset_kwargs
from octo.data.utils.data_utils import get_dataset_statistics


def generate_data_statistics(config):
    tf.config.set_visible_devices([], "GPU")
    if (
        standardize_fn := config.get("standardize_fn", None)
    ) is not None:

        if isinstance(standardize_fn, str):
            path, f_name = standardize_fn.split(":")
            # imp is deprecated, but it's also what ml_collections uses
            standardize_fn = getattr(imp.load_source("standardize_fn", path), f_name)

            del config["standardize_fn"]
            config["standardize_fn"] = standardize_fn

        elif isinstance(standardize_fn, types.FunctionType):
            standardize_fn = config["standardize_fn"]

        else:
            raise ValueError

    ds_name = config["name"]
    data_dir = config["data_dir"]
    restructure_fn = partial(
        restructure,
        name=ds_name,
        standardize_fn=standardize_fn,
        image_obs_keys=config.get("image_obs_keys", {}),
        depth_obs_keys=config.get("depth_obs_keys", {}),
        state_obs_keys=config.get("state_obs_keys", []),
        language_key=config.get("language_key", None),
        absolute_action_mask=config.get("absolute_action_mask", []),
    )
    print(ds_name, data_dir)
    builder = tfds.builder(ds_name, data_dir=data_dir)

    # load or compute dataset statistics
    full_dataset = dl.DLataset.from_rlds(
        builder, split="all", shuffle=False, num_parallel_reads=tf.data.AUTOTUNE
    ).traj_map(restructure_fn, tf.data.AUTOTUNE)
    # tries to load from cache, otherwise computes on the fly
    get_dataset_statistics(
        full_dataset,
        hash_dependencies=(
            str(builder.info),
            str(config.get("state_obs_keys", [])),
            inspect.getsource(standardize_fn) if standardize_fn is not None else "",
        ),
        save_dir='./',
    )
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_path", type=str, required=True, help="Name of the prior dataset to generate statistics for")
    args = parser.parse_args()

    config = dataset_kwargs.copy()
    config["name"] = os.path.basename(args.prior_path)
    config["data_dir"] = os.path.dirname(args.prior_path)

    generate_data_statistics(config)