import jax
import numpy as np
from flatten_dict import flatten, unflatten
from streaming import StreamingDataset

from typing import Callable, Any

from absl import flags
import datamil.metagradients.flags_config as flags_config
FLAGS = flags.FLAGS

def slice_batch(batch, sel):
    if isinstance(batch, dict):
        return {k: slice_batch(v, sel) for k, v in batch.items()}
    else:  # numpy array case
        return batch[sel]

class RoboDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable=None,
                 **kwargs,
                ) -> None:
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=batch_size,
            **kwargs,
        )
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        item = super().__getitem__(idx)
        item = unflatten(item, 'dot')

        if self.transforms is None:
            return flatten(item, 'dot')
        else:
            return flatten(self.transforms(item), 'dot')
        
class REPLAYBatch:
    def __init__(self, bs):
        # batch size of this batch
        self.bs = bs

    def get_minibatches(self, part):
        # part: 'train', 'val', or 'meta'
        raise NotImplementedError

class REPLAYMinibatches:
    def __init__(self, bs):
        self.bs = bs

    def __iter__(self):
        # iterates over minibatches, each minibatch is a (ixs, (x, y)) tuple
        # where ixs, x and y are jax arrays or numpy arrays
        # ixs: indices of the data points in the minibatch
        # x: input data
        # y: output data
        raise NotImplementedError

class RoboREPLAYBatch(REPLAYBatch):
    def __init__(self,
                 batch: dict,
                 bs: int,
                 minibs: int,
                 sharding: str,
                 state=None,
                 batch_idx:int=0,
                 offset:int=0):

        assert bs % minibs == 0, "Minibatch size does not divise batch size"

        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.sharding = sharding
        self.state = state
        self.batch_idx = batch_idx
        self.offset = offset

    def get_state(self):
        return self.state

    def get_batch_idx(self):
        return self.batch_idx

    def set_offset(self, offset):
        self.offset = offset

    def get_minibatches(self, part: str):
        if self.sharding is not None:
            batch = jax.device_put(self.batch, self.sharding)
        else:
            batch = self.batch
        minibs = int({
            # 'train': 1,
            # 'val': 1,
            'train': 0.5,
            'val': 0.5,
            'meta': 0.5
        }[part] * self.minibs)

        return RoboREPLAYMinibatches(self.bs, batch, minibs=minibs, offset=self.offset)

class RoboREPLAYMinibatches(REPLAYMinibatches):
    def __init__(self,
                 bs: int,
                 batch: dict,
                 minibs: int,
                 offset: int=0):

        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.offset = offset

    def make_iterator(self, batch, minibs):
        s = 0
        this_bs = batch['index'].shape[0]
        while True:
            e = min(s + minibs, this_bs)
            sel = slice(s, e)
            s = e

            mini_batch = slice_batch(batch, sel)

            indices = mini_batch.get('index', None) + self.offset
            if 'index' in mini_batch:
                mini_batch['index'] += self.offset
            y = None
            yield indices, (mini_batch, y)

            if e == this_bs:
                break

    def __iter__(self):
        return self.make_iterator(self.batch, self.minibs)
    
class SpecialReplayBatch(REPLAYBatch):
    """
    Similar to RoboREPLAYBatch, but under the hood it doesn't load the entire batch.
    Instead, it will stream from data_iter chunk by chunk in get_minibatches(...).
    """

    def __init__(
        self,
        data_iter,
        global_iter: int,
        global_seed: int,
        batch_size: int,
        iter_bs: int,
        minibs: int,
        sharding,
        offset: int
    ):
        # Like RoboREPLAYBatch, we call super().__init__(bs)
        super().__init__(batch_size)

        self.data_iter = data_iter
        self.global_iter = global_iter
        self.global_seed = global_seed
        # self.bs = batch_size
        self.iter_bs = iter_bs
        self.minibs = minibs
        self.sharding = sharding
        self.offset = offset

        self.batch_idx = FLAGS.config.num_steps - FLAGS.config.bob_steps

        # We yield ourselves exactly once, just like your original design
        self._exhausted = False

    def __len__(self):
        # If data_iter is sized, you can do len(self.data_iter),
        # otherwise remove or approximate
        return len(self.data_iter)

    def __iter__(self):
        """So `for x in special_batch:` yields exactly one x (self)."""
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        self._exhausted = True
        return self

    def get_minibatches(self, part: str):
        """
        Instead of yielding directly, we return a SpecialReplayMinibatches
        object (like RoboREPLAYBatch returns RoboREPLAYMinibatches).
        That object will handle chunked iteration under the hood.
        """
        return SpecialReplayMinibatches(
            bs=self.bs,
            iter_bs=self.iter_bs,
            data_iter=self.data_iter,
            global_iter=self.global_iter,
            global_seed=self.global_seed,
            minibs=self.minibs,
            offset=self.offset,
            sharding=self.sharding,
            batch_idx=self.batch_idx,
            part=part
        )

class SpecialReplayMinibatches(REPLAYMinibatches):
    """
    Similar to RoboREPLAYMinibatches, but it streams multiple 512-chunks from `data_iter`.
    Each chunk is turned into an RoboREPLAYBatch, and we yield from that batch's minibatches.
    """

    def __init__(
        self,
        bs: int,
        iter_bs: int,
        data_iter,
        global_iter,
        global_seed,
        minibs: int,
        offset: int,
        sharding,
        batch_idx: int,
        part: str
    ):
        # Like RoboREPLAYMinibatches, we call super().__init__(bs).
        super().__init__(bs)

        self.data_iter = data_iter
        self.global_iter = global_iter
        self.global_seed = global_seed
        self.minibs = minibs
        self.offset = offset
        self.sharding = sharding
        self.batch_idx = batch_idx
        self.part = part

        # This object can have .bs, .minibs, etc. that calling code might expect
        self.bs = bs
        self.iter_bs = iter_bs

    def __iter__(self):
        """
        We'll fetch each 512-sized chunk from `self.data_iter`, create an RoboREPLAYBatch,
        then yield all sub-minibatches from that batch.
        """
        for chunk in self.data_iter:
            # Possibly check if chunk is empty (shape[0] == 0). If so, skip or break
            first_key = next(iter(chunk.keys()))
            if chunk[first_key].shape[0] == 0:
                break

            # Build seed array
            seed = (self.global_iter * self.iter_bs) \
                   + np.arange(self.iter_bs) \
                   + int(self.global_seed * 1e9)
            seed = seed.astype('int64')
            chunk['seed'] = seed

            # unflatten if needed
            chunk = unflatten(chunk, 'dot')

            # Wrap in an RoboREPLAYBatch (just like your normal pipeline)
            replay_batch = RoboREPLAYBatch(
                batch=chunk,
                bs=self.iter_bs,
                minibs=self.minibs,
                sharding=self.sharding,
                state=self.data_iter.state_dict() if hasattr(self.data_iter, 'state_dict') else None,
                batch_idx=self.batch_idx
            )

            if self.offset:
                replay_batch.set_offset(1_000_000)

            # Increase global_iter for the next chunk
            self.global_iter += 1

            # "Yield from" the sub-minibatches of this chunk
            yield from replay_batch.get_minibatches(self.part)