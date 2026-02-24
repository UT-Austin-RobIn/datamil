import imp
import os
import types
import json

import jax
import jax.numpy as jnp
import numpy as np

import flax
import tensorflow as tf
from absl import app, flags
from flatten_dict import unflatten

from octo.utils.spec import ModuleSpec
from datamil.metagradients.core.dataloading import RoboDataset, RoboREPLAYBatch, SpecialReplayBatch

from streaming import StreamingDataLoader

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

import datamil.metagradients.flags_config as flags_config
FLAGS = flags.FLAGS

TEXT_PROCESSOR_CONFIG = {
    'args': [], 
    'kwargs': {
        'encode_with_model': False, 
        'tokenizer_kwargs': {
            'max_length': 16, 
            'padding': 
            'max_length', 
            'return_tensors': 'np', 
            'truncation': True
        }, 
        'tokenizer_name': 't5-base'}, 
        'module': 'octo.data.utils.text_processing', 
        'name': 'HFTokenizer'
}
if TEXT_PROCESSOR_CONFIG is None:
    text_processor = None
else:
    text_processor = ModuleSpec.instantiate(TEXT_PROCESSOR_CONFIG)()

def process_item(item):
    def process_text(item, text_processor):
        if text_processor is None:
            item["task"].pop("language_instruction")

        else:
            item["task"]["language_instruction"] = text_processor.encode(
                item["task"]["language_instruction"].decode("utf-8")
            )
            item["task"]["language_instruction"]["input_ids"] = item["task"]["language_instruction"]["input_ids"].squeeze()
            item["task"]["language_instruction"]["attention_mask"] = item["task"]["language_instruction"]["attention_mask"].squeeze()

        return item

    item = process_text(item, text_processor)
    del item["dataset_name"]
    return item

def make_replay_dataset(start_batch: int,
                        end_batch: int,
                        sharding: str,
                        train: bool=True,
                        return_dw_only: bool=False,
                        index_path: str=''):
    
    if return_dw_only:
        data_weights = jax.numpy.ones((1_000_000,), dtype=jnp.float32)
        return None, data_weights

    # initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    if train:
        mds_path = os.path.join(
            FLAGS.config.dataset_kwargs.data_dir,
            FLAGS.config.dataset_kwargs.name,
        )
    else:
        mds_path = os.path.join(
            FLAGS.config.val_dataset_kwargs.data_dir,
            FLAGS.config.val_dataset_kwargs.name,
        )

    if train:
        batch_size = FLAGS.config.batch_size
    else:
        batch_size = FLAGS.config.val_batch_size

    # dataset = StreamingDataset(
    dataset = RoboDataset(
        local=mds_path,
        remote=None,
        shuffle=train,
        shuffle_seed=FLAGS.config.seed,
        batch_size=batch_size,
        transforms=process_item,
        index_filename=index_path
    )

    def numpy_collate(batch):
        return {
            k: np.array([sample[k] for sample in batch], dtype=object if isinstance(batch[0][k], bytes) else None)
            for k in batch[0].keys()
        }

    dataloader = StreamingDataLoader(
        dataset,
        drop_last=train,
        batch_size=batch_size,
        num_workers=FLAGS.config.num_workers,
        prefetch_factor=2,  # Optional: controls samples prefetched per worker
        collate_fn=numpy_collate,
    )

    current_epoch, remaining_batches = divmod(start_batch, len(dataloader))
    remaining_samples = remaining_batches * batch_size

    state = {
        'epoch': current_epoch,
        'sample_in_epoch': remaining_samples,
        'num_canonical_nodes': 1,
        'shuffle_seed': FLAGS.config.seed,
        'initial_physical_nodes': 1
    }

    dataloader.load_state_dict(state)

    data_weights = jax.numpy.ones((1_000_000,), dtype=jnp.float32)

    return dataloader, data_weights

def make_special_dataset(start_batch: int,
                         end_batch: int,
                         sharding: str,
                         return_dw_only: bool=False):

    # initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    mds_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
    )

    checkpoint_path = FLAGS.config.checkpoint_path
    special_index_path = os.path.join(
        checkpoint_path,
        'special_index.json'
    )

    with open(special_index_path, 'r') as f:
        index = json.load(f)

    batch_size = FLAGS.config.batch_size

    # dataset = StreamingDataset(
    dataset = RoboDataset(
        local=mds_path,
        remote=None,
        shuffle=True,
        shuffle_seed=FLAGS.config.seed,
        batch_size=batch_size,
        transforms=process_item,
        index_filename=special_index_path
    )

    def numpy_collate(batch):
        return {
            k: np.array([sample[k] for sample in batch], dtype=object if isinstance(batch[0][k], bytes) else None)
            for k in batch[0].keys()
        }

    dataloader = StreamingDataLoader(
        dataset,
        drop_last=True,
        batch_size=batch_size,
        num_workers=FLAGS.config.num_workers,
        prefetch_factor=2,  # Optional: controls samples prefetched per worker
        collate_fn=numpy_collate,
    )

    return dataloader, None

def make_replay_iterators(
        start_batch,
        end_batch,
        sharding,
        data_iterators,
        mode,
        global_seed,
        batch_size
    ):

    ds_iter, ds_attrib = data_iterators
    if mode != 'train':
        assert ds_attrib is None

    # start returning batches
    batch_idx = start_batch

    if mode == 'train':
        minibs = FLAGS.config.mini_batch_size
    else:
        minibs = FLAGS.config.mini_val_batch_size

    special_batch = FLAGS.config.num_steps - FLAGS.config.bob_steps

    if batch_idx <= special_batch:
        global_iter = batch_idx
    else:
        global_iter = start_batch + len(ds_attrib)

    if ds_attrib is not None:
        ds_special = SpecialReplayBatch(
            data_iter=ds_attrib,
            global_iter=global_iter,
            global_seed=global_seed,
            batch_size=batch_size*len(ds_attrib),
            iter_bs=batch_size,
            minibs=minibs,
            sharding=sharding,
            offset=1_000_000,
        )
    else:
        ds_special = None

    while batch_idx < end_batch:

        if mode != 'train':
            data_iterator = ds_iter
        else:
            if batch_idx == special_batch:
                data_iterator = ds_special
            else:
                data_iterator = ds_iter

        print('\n\n')
        print('*'*20)
        print(mode, start_batch, end_batch, len(data_iterator))
        assert mode=='train' or len(data_iterator) == end_batch
        print('*'*20)
        total_samples = 0
        for batch in data_iterator:
            if batch_idx >= end_batch:
                break

            if batch_idx == special_batch:
                yield ds_special

            else:
                # here it's the regular iterator
                # we keep the same
                seed = global_iter * batch_size + np.arange(batch_size) + global_seed * 1e9
                seed = seed.astype('int64')
                batch['seed'] = seed

                batch = unflatten(batch, 'dot')

                replay_batch = RoboREPLAYBatch(
                    batch=batch,
                    bs=batch_size,
                    minibs=minibs,
                    sharding=sharding if mode=='train' else None,
                    state=data_iterator.state_dict(),
                    batch_idx=batch_idx,
                )

                yield replay_batch

                global_iter += 1

            batch_idx += 1
            total_samples += batch_size
        print(mode, f'total_samples: {total_samples}')
        print('*'*20, '\n\n')

        # what to do here?
        # if mode == 'train' and batch_idx == special_batch and not entering_special:
        #     global_iter += 1
        #     batch_idx += 1

def create_special_index(iter_seed:int=0):

    checkpoint_path = FLAGS.config.checkpoint_path
    special_index_path = os.path.join(
        checkpoint_path,
        'special_index.json'
    )

    # if os.path.exists(special_index_path):
    #     return

    index_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
        'index.json'
    )

    with open(index_path, 'r') as f:
        index = json.load(f)

    num_candidate_shards = int(
        FLAGS.config.candidate_size * len(index['shards'])
    )

    rng = np.random.default_rng(iter_seed + FLAGS.config.seed)
    candidate_shards = rng.choice(index['shards'], num_candidate_shards, replace=False).tolist()

    special_index = {
        'shards': candidate_shards,
        'version': 2
    }

    with open(special_index_path, 'w') as f:
        json.dump(special_index, f)

def make_split_loader_and_data_weights(start_batch: int,
                                       end_batch: int,
                                       sharding: str,
                                       mode: str='train',
                                       seed: int=42,
                                       iter_seed: int=0):

    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        create_special_index(iter_seed=iter_seed)

    # ds_iter, _ = make_replay_dataset(start_batch, end_batch, sharding, train=(mode=='train'), index_path=FLAGS.config.include_index_path)
    if mode == 'train':
        ds_iter, _ = make_replay_dataset(start_batch, end_batch, sharding, train=True, index_path=FLAGS.config.include_index_path)
        ds_attrib, _ = make_special_dataset(start_batch, end_batch, sharding)
    else:
        ds_iter, _ = make_replay_dataset(start_batch, end_batch, sharding, train=False)
        ds_attrib = None

    if mode == 'train':
        batch_size = FLAGS.config.batch_size
    else:
        batch_size = FLAGS.config.val_batch_size

    # return make_replay_iterators(start_batch, end_batch, sharding, ds_iter, global_seed=seed, batch_size=FLAGS.config.batch_size)
    return make_replay_iterators(
        start_batch,
        end_batch,
        sharding,
        (ds_iter, ds_attrib),
        mode=mode,
        global_seed=seed,
        batch_size=batch_size
    )