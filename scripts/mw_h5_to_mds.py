import os
import json

import numpy as np


from pathlib import Path
from copy import deepcopy


from flatten_dict import flatten

from streaming import MDSWriter
from tqdm import tqdm

columns = {
    'action': 'ndarray:float32',
    'observation': 'ndarray:float32',
    'index': 'int64',
}

target_dtypes = {
    k: v.replace('ndarray:', '')
    for k, v in columns.items()
}

import glob
import os
import h5py
class MetaworldDataset:

    def __init__(self, data_folder):
        super().__init__()

        self.hdf5_files = sorted(glob.glob(os.path.join(data_folder, "**", "*.h5"), recursive=True))
        
        self.actions = []
        self.observations = []
        self.index = []
        for i, file in tqdm(enumerate(self.hdf5_files)):
            with h5py.File(file, 'r') as f:
                self.actions.extend(f['actions'][:])
                self.observations.extend(f['observations'][:])
                self.index.extend([i] * len(f['actions'][:]))

        # Convert to numpy arrays
        self.actions = np.array(self.actions)
        self.observations = np.array(self.observations)
        self.index = np.array(self.index)
        
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        observation = self.observations[idx]

        return {'action': action, 'observation': observation, 'index': self.index[idx]}

def make_dataset(path):
    return MetaworldDataset(path)

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path of the dataset to convert')
    parser.add_argument('--out_root', type=str, default=None, help='Output path for the MDS dataset')
    args = parser.parse_args()

    dataset = make_dataset(args.dataset_path)

    target_size = 150 * 1024 * 1024 # 130MB
    item_size = 640 # 640B
    num_items = np.ceil(target_size / item_size)
    shard_size = int(num_items * item_size)

    out_path = os.path.join(args.out_root, args.dataset_path.split('/')[-1] + '_mds')

    os.makedirs(out_path, exist_ok=True)

    prev_index = None
    writer = None
    for item in tqdm(dataset):
        index = item['index']

        if index != prev_index:
            path = os.path.join(out_path, f"traj_{index}")
            assert not os.path.exists(path), f"File {path} already exists"
            if writer is not None:
                writer.finish()
            writer = MDSWriter(columns=columns, out=path, size_limit=shard_size)
            prev_index = index
        
        item_flat = {}
        for k, v in flatten(item, 'dot').items():
            item_flat[k] = v.astype(target_dtypes[k])
        writer.write(item_flat)

    writer.finish()
    create_unified_index(out_path)


if __name__ == "__main__":
    main()