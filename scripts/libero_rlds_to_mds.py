import imp
import os
import types
import json

import numpy as np
import tensorflow as tf

from datamil.data.dataset import make_single_dataset
from pathlib import Path
from copy import deepcopy
from flatten_dict import flatten

from streaming import MDSWriter
from tqdm import tqdm

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

traj_transform_kwargs = dict(
        window_size=1,
        future_action_window_size=7,
        goal_relabeling_strategy=None,
        task_augment_strategy=None,
        task_augment_kwargs=dict(
            keep_image_prob=0.0,
        ),
    )
frame_transform_kwargs = dict(
    resize_size={
        "primary": (128, 128),  # workspace (3rd person) camera is at 256x256
        "wrist": (128, 128),  # wrist camera is at 128x128
    },
    image_augment_kwargs=dict(),
)


columns = {
    'absolute_action_mask': 'ndarray:uint8',
    'action': 'ndarray:float32',
    'dataset_name': 'bytes',

    # 'weights': 'ndarray:float32',
    'index': 'int64',

    'observation.image_primary': 'ndarray:uint8',
    'observation.image_wrist': 'ndarray:uint8',
    'observation.pad_mask': 'ndarray:uint8',
    'observation.pad_mask_dict.image_primary': 'ndarray:uint8',
    'observation.pad_mask_dict.image_wrist': 'ndarray:uint8',
    'observation.pad_mask_dict.proprio': 'ndarray:uint8',
    'observation.pad_mask_dict.timestep': 'ndarray:uint8',
    'observation.proprio': 'ndarray:float32',
    'observation.timestep': 'ndarray:int32',
    'task.language_instruction': 'bytes',

    # 'task.pad_mask_dict.language_instruction': 'ndarray:uint8'
    'task.pad_mask_dict.language_instruction': 'uint8'
}

target_dtypes = {
    k: v.replace('ndarray:', '')
    for k, v in columns.items()
}

def make_replay_dataset(config):

    # initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")
    if (standardize_fn := config.get("standardize_fn", None)) is not None:

        if isinstance(standardize_fn, str):
            path, name = standardize_fn.split(":")
            # imp is deprecated, but it's also what ml_collections uses
            standardize_fn = getattr(imp.load_source("standardize_fn", path), name)

            del config["standardize_fn"]
            config["standardize_fn"] = standardize_fn

        elif isinstance(standardize_fn, types.FunctionType):
            standardize_fn = config["standardize_fn"]

        else:
            raise ValueError

    ############################################
    ########## get tfrecords iterator ##########
    ############################################

    print('Creating dataset')
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(0)
    # create dataset object
    dataset = make_single_dataset(
        config,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=False,
        shuffle=False,
        num_parallel_calls=1,
        num_parallel_reads=1,
    )
    return dataset

def create_unified_index(dataset_dir):
    dataset_dir = Path(dataset_dir)

    indices_json = list(dataset_dir.rglob('index.json'))
    parent_json_path = dataset_dir / 'index.json'

    parent_data = {
        'shards': [],
        'version': 2
    }

    for json_path in tqdm(indices_json):
        assert json_path.exists()

        with open(json_path, 'r') as f:
            data_child = json.load(f)

        assert data_child.keys() == parent_data.keys()

        for sample in data_child['shards']:
            correct_path_sample = deepcopy(sample)

            new_dir = str(json_path).replace('/index.json', '').replace(
                str(dataset_dir),
                ''
            )[1:]

            old_raw_data_path = sample['raw_data']['basename']
            new_raw_data_path = os.path.join(new_dir, old_raw_data_path)
            correct_path_sample['raw_data']['basename'] = new_raw_data_path

            if sample['zip_data'] is not None:
                old_zip_data_path = sample['zip_data']['basename']
                new_zip_data_path = os.path.join(new_dir, old_zip_data_path)
                correct_path_sample['zip_data']['basename'] = new_zip_data_path

            parent_data['shards'].append(deepcopy(correct_path_sample))
            del correct_path_sample

    parent_json_path = dataset_dir / 'index.json'

    if parent_json_path.exists():
        raise FileExistsError('Delete the old `index.json` file then re-run the code')

    with open(parent_json_path, 'w') as f:
        json.dump(parent_data, f)


def main(config, out_path):
    ds_name = config['name']
    ds_path = config['data_dir'] 

    print('dataset:', ds_name)
    print('path:', ds_path)

    dataset = make_replay_dataset(config)
    dataset_statistics = dataset.dataset_statistics
    dataset = dataset.unbatch().iterator()

    target_size = 150 * 1024 * 1024 # 130MB
    item_size = 640 # 640B
    num_items = np.ceil(target_size / item_size)
    shard_size = int(num_items * item_size)

    out_path = os.path.join(out_path, ds_name)
    os.makedirs(out_path, exist_ok=True)

    prev_index = None
    writer = None
    for item in tqdm(dataset):

        index = item['index'][0]

        if index != prev_index:
            path = os.path.join(out_path, f"traj_{index}")
            assert not os.path.exists(path), f"File {path} already exists"
            if writer is not None:
                writer.finish()
            writer = MDSWriter(columns=columns, out=path, size_limit=shard_size)
            prev_index = index
        
        item_flat = {}
        for k, v in flatten(item, 'dot').items():
            if k in ['dataset_name', 'task.language_instruction']:
                item_flat[k] = v
            else:
                item_flat[k] = v.astype(target_dtypes[k])

        writer.write(item_flat)

    def numpy_to_list(obj):
        if isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    dataset_statistics = numpy_to_list(dataset_statistics)
    json_path = os.path.join(out_path, 'dataset_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_statistics, f, indent=4)
    writer.finish()

    create_unified_index(out_path)


if __name__ == "__main__":
    import argparse
    from datamil import libero_dataset_kwargs as dataset_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the output directory where MDS clusters will be saved")
    args = parser.parse_args()
    
    config = dataset_kwargs.copy()
    config['data_dir'] = os.path.dirname(args.data_path)
    config['name'] = os.path.basename(args.data_path)

    main(config, args.out_path)