from typing import Iterator, Tuple, Any
import os
import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import cv2
from tqdm import tqdm
import hashlib

DATA_DIR = os.environ['DATA_DIR']
DATASET_NAME = os.environ['TASK_NAME']
NUM_DEMOS = 5

def seed_np_with_string(seed_string):
    # Convert the string to a hash
    hash_value = hashlib.sha256(seed_string.encode()).digest()

    # Convert the hash to an integer
    seed_int = int.from_bytes(hash_value, byteorder='big')
    seed_int = seed_int % (2**31 - 1)
    
    # Set the seed for NumPy
    np.random.seed(seed_int)

def upweight_grasp_states(actions):
    gripper = actions[:, -1]

    weights = np.ones(gripper.shape, dtype=np.float32)
    for j in range(len(weights)-1):
        # if gripper[j]==1 and gripper[j+1]==0:
        if (gripper[j]==1 and gripper[j+1]==0) or (gripper[j]==0 and gripper[j+1]==1):
            weights[j-19:j+1] = 2

    return weights

class LiberoVal(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
      '0.1.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self.index = 0

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot eef position and quaternion, '
                                '1x gripper openness].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x cartesian position, 3x cartesian rotation,'
                            '1x gripper open/close.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'index': tfds.features.Tensor(
                        shape=(1,),
                        dtype=np.int32,
                        doc='Index of the trajectory in the dataset.'
                    ),
                    'weights': tfds.features.Tensor(
                        shape=(1,),
                        dtype=np.float32,
                        doc='Weights for the actions, used for training.'
                    )
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'task_name': tfds.features.Text(
                        doc='Name of the task.'
                    ),
                    'demo_id': tfds.features.Text(
                        doc='Unique ID of the demo.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(source_path=os.path.join(DATA_DIR, "libero_10", DATASET_NAME + '_demo.hdf5'), train=True),
            # 'val': self._generate_examples(source_path=os.path.join(DATA_DIR, "libero_10", DATASET_NAME + '_demo.hdf5'), train=False),
        }

    def _generate_examples(self, source_path, train) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(demo_id, task_name):
            language_instruction = task_name.split('SCENE')[1][2:]
            language_instruction = " ".join(language_instruction.split('_'))
            print(self.index, language_instruction, '-', demo_id)

            # load raw data --> this should change for your dataset
            data = demo_data[demo_id]  # this is a list of dicts in our case
            
            actions = data['actions'][:].astype(np.float32)
            print(actions.shape)
            actions[:, -1] = (1 - actions[:, -1])/2

            weights = np.ones(actions.shape[0], dtype=np.float32) # upweight_grasp_states(actions)
            # print('upweighted states', np.sum(weights>1.5))

            data_len = actions.shape[0]

            primary_image = np.flip(data['obs']['agentview_rgb'][:].astype(np.uint8), axis=1)
            wrist_image = np.flip(data['obs']['eye_in_hand_rgb'][:].astype(np.uint8), axis=1)
            ee_pos = data['obs']['ee_pos'][:].astype(np.float32)
            ee_ori = data['obs']['ee_ori'][:].astype(np.float32)
            gripper_states = data['obs']['gripper_states'][:][:, :1].astype(np.float32)
            states = np.concatenate([ee_pos, ee_ori, np.zeros((data_len, 1), dtype=np.float32), gripper_states], axis=-1)

            episode = []
            # language_embedding = self._embed([language_instruction])[0].numpy()
            for i in range(data_len):
                # compute Kona language embedding
                
                episode.append({
                    'observation': {
                        'image': primary_image[i],
                        'wrist_image': wrist_image[i],
                        'state': states[i],
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'reward': float(i == (data_len - 1)),
                    'is_first': i == 0,
                    'is_last': i == (data_len - 1),
                    'is_terminal': i == (data_len - 1),
                    'language_instruction': language_instruction,
                    'index': np.array([self.index], dtype=np.int32),
                    'weights': np.array([weights[i]], dtype=np.float32)
                })

            # cv2.imwrite('im.png', episode[0]['observation']['image'])
            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'demo_id': demo_id,
                    'task_name': task_name,
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return self.index, sample

        
        
        # create list of all examples
        f = h5py.File(source_path, 'r')
        demo_data = f['data']

        demo_keys = sorted(list(demo_data.keys()))
        
        seed_np_with_string(DATASET_NAME)
        demo_keys = np.random.permutation(demo_keys)
        if train:
            demo_keys = demo_keys[:NUM_DEMOS] # [:10]
        else:
            demo_keys = demo_keys[NUM_DEMOS:]

        task_name = DATASET_NAME
        for demo_id in demo_keys:
            yield _parse_example(demo_id, task_name)
            self.index += 1

        f.close()
