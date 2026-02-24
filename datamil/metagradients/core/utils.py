import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import os
import dill
import weakref
import shutil
from functools import partial
from pathlib import Path
from functools import cache

def set_dtype(dty, determinism):
    if dty == 'float64':
        jax.config.update('jax_enable_x64', True)
        raise ValueError('float64 not supported')
    elif dty == 'float32':
        jax.config.update('jax_default_matmul_precision', 'float32')
        raise ValueError('float32 not supported')
    elif dty == 'tf32':
        jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
        return
    else:
        raise ValueError(f'Unknown dtype {dty}')

@cache
def make_shardings():
    num_devices = len(jax.devices('gpu'))
    if num_devices > 1:
        print('MAKING SHARDINGS FOR SINGLE DEVICE')
        mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), 'batch')
        sharding = NamedSharding(mesh, P('batch'))
        replicated_sharding = NamedSharding(mesh, P())
        return sharding, replicated_sharding
    else:
        print('MAKING SHARDINGS FOR SINGLE DEVICE')
        sharding = jax.devices('gpu')[0]
        return sharding, sharding

class NotANumberError(Exception):
    pass

def wipe_rep(x):
    x.__repr__ = lambda: 'per_sample_loss'
    x.__str__ = lambda: 'per_sample_loss'
    return x

def pytree_size(v):
    def get_size(x):
        if hasattr(x, 'nbytes'):
            return x.nbytes
        else:
            return 0

    sizes = jax.tree_util.tree_map(get_size, v)
    tree_sum = jax.tree_util.tree_reduce(lambda x, y: (x + y), sizes, 0)
    return tree_sum * 1e-9

@jax.jit
def pytree_gbs(pytree):
    flat, _ = jax.tree_util.tree_flatten(pytree)
    total_size_in_bytes = sum(arr.nbytes for arr in flat)
    return total_size_in_bytes * 1e-9

@jax.jit
def safe_zeros(x):
    if isinstance(x, int) or x.dtype == jax.float0:
        return x

    return jnp.zeros_like(x)

@jax.jit
def safe_divide(x, y):
    if isinstance(x, int) or x.dtype == jax.float0 or x.dtype == jnp.int32 or x is None or y is None:
        return x

    return x / y

@jax.jit
def safe_diff(x, y):
    if x is None or x.dtype == jax.float0 or isinstance(x, int):
        return x

    return jnp.abs(x - y)

@jax.jit
def safe_add(x, y):
    if hasattr(x, 'dtype'):
        if x.dtype == jax.float0:
            return x

    if x is None:
        return None

    if y is None:
        return x

    try:
        res = x + y
    except:
        import pdb; pdb.set_trace()

    return res

# @partial(jax.jit, donate_argnums=(1,))
@partial(jax.jit)
def add_trees_ignore_none(x, y):
    def safe_add(x, y):
        if x.dtype == jax.float0:
            return x

        if x is None:
            return None

        if y is None:
            return x

        try:
            res = x + y
        except:
            import pdb; pdb.set_trace()

        return res

    # if x None then the acc is nothing so we just return y
    if x is None:
        return y

    return jax.tree_util.tree_map(safe_add, x, y)

# @partial(jax.jit, donate_argnums=(1,))
@partial(jax.jit)
def add_trees(x, y):
    def safe_add(x, y):
        if x.dtype == jax.float0:
            return x

        if x is None or y is None:
            return None

        try:
            res = x + y
        except:
            import ipdb; ipdb.set_trace()

        return res

    return jax.tree_util.tree_map(safe_add, x, y)

@cache
def get_one():
    return jnp.ones((), dtype=jnp.float32)

class MinibatchedLoader():
    def __init__(self, num_batches, num_datapoints):
        self.num_batches = num_batches
        self.num_datapoints = num_datapoints

    # map from (start_batch, end_batch, sharding) -> Iterator[MiniBatchIterator]
    #    represents the minibatches from start_batch to end_batch (exclusive at end)
    def make_batch_iterator(self, start_batch, end_batch, sharding):
        raise NotImplementedError

@jax.jit
def safe_tree_add(x, y):
    return jax.tree_util.tree_map(safe_add, x, y)

class MiniBatchIterator:
    # minibatches: iterator of minibatches
    # length: total number of datapoints
    def __init__(self, minibatches, length):
        self.minibatches = minibatches
        self.length = length
        self.iterator = iter(minibatches)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def __len__(self):
        return self.length

def cleanup_path(p):
    print('Cleaning up path', p)
    shutil.rmtree(p)

class DiskBackedDict:
    def __init__(self, path):
        self.path = Path(path)
        assert self.path.exists()
        assert self.path.is_dir()
        self.backend = {}
        cleaner = partial(cleanup_path, p=path)
        weakref.finalize(self, cleaner)

    def __setitem__(self, key, value):
        out_path = self.path / f'{key}'
        self.backend[key] = out_path
        with open(out_path, 'wb') as f:
            dill.dump(value, f)

    def __contains__(self, key):
        return key in self.backend

    def __getitem__(self, key):
        p = self.backend[key]
        assert isinstance(p, Path)
        with open(p, 'rb') as f:
            return dill.load(f)

    def __len__(self):
        return len(self.backend)

    def keys(self):
        return self.backend.keys()

    def __delitem__(self, key):
        del self.backend[key]
        os.remove(self.path / f'{key}')

class CastedDict:
    def __init__(self, cast_state, check_count=False, disk_path=None):
        self.cast = cast_state
        self.check_count = check_count
        self.backend = {}
        self.disk_backend = DiskBackedDict(disk_path) if disk_path is not None else {}
        self.disk_path = disk_path

    def __contains__(self, key):
        return key in self.backend or key in self.disk_backend

    def __getitem__(self, key):
        if key in self.backend:
            v = self.backend.__getitem__(key)
        elif key in self.disk_backend:
            v = self.disk_backend.__getitem__(key)
        else:
            raise KeyError(key)

        v = self.cast.replace(params=v.params, opt_state=v.opt_state)
        if self.check_count:
            opt_count = v.opt_state.count
            assert opt_count == key, (opt_count, key)

        return v

    def keys(self):
        all_keys = list(self.backend.keys()) + list(self.disk_backend.keys())
        return all_keys

    def __len__(self):
        return len(self.backend) + len(self.disk_backend)

    def __delitem__(self, key):
        if key in self.backend:
            del self.backend[key]
        elif key in self.disk_backend:
            del self.disk_backend[key]
        else:
            raise KeyError(key)

    def set(self, key, value, disk=False):
        if self.check_count:
            opt_count = value.opt_state.count
            assert opt_count == key, (opt_count, key)

        if not disk:
            res = self.backend.__setitem__(key, value)
        else:
            assert self.disk_path is not None
            res = self.disk_backend.__setitem__(key, value)

        return res

def gpu_show():
    if not os.environ.get('SSH_TTY', False):
        print('>> GPU', os.environ['CUDA_VISIBLE_DEVICES'])
        print('>> MACHINE HOSTNAME', os.environ.get('HOSTNAME', 'no host'))
        os.system('nvidia-smi')

def pytree_diff_stats(pytree1, pytree2):
    pytree1 = pytree2.replace(params=pytree1.params, opt_state=pytree1.opt_state)
    return _pytree_diff_stats(pytree1, pytree2)

def _pytree_diff_stats(pytree1, pytree2):
    diffs = jax.tree_util.tree_map(safe_diff, pytree1, pytree2)
    flat_diffs, _ = tree_flatten(diffs)

    flat_diffs_concat = jnp.concatenate([jnp.ravel(diff) for diff in flat_diffs])
    max_diff = jnp.max(flat_diffs_concat)
    avg_diff = jnp.mean(flat_diffs_concat)
    std_diff = jnp.std(flat_diffs_concat)

    return max_diff, avg_diff, std_diff