import time
from functools import partial
import jax
import jax.numpy as jnp
import cupy
import numpy as np

def make_io_stream():
    return cupy.cuda.Stream(non_blocking=True)

def is_cpu(x):
    return list(x.devices())[0].platform == 'cpu'

def _dlpack_gpu2cpu(x, out=None, stream=None):
    if isinstance(x, int) or x.dtype == jax.float0:
        return x

    if isinstance(x, np.ndarray) or is_cpu(x):
        return x

    assert stream is not None

    # x = jax.dlpack.to_dlpack(x, copy=False)
    x = jax.dlpack.to_dlpack(x)
    x = cupy.from_dlpack(x)

    if out is None:
        out = cupy.cuda.alloc_pinned_memory(x.nbytes)
        out = np.frombuffer(out, x.dtype, x.size).reshape(x.shape)

    assert out.shape == x.shape
    assert out.dtype == x.dtype

    x = cupy.asnumpy(x, order='A', blocking=False, out=out, stream=stream)
    return x

class NoGCVessel:
    def __init__(self, x_jax, x_cpu, x_cupy, x_dlpack, stream):
        self.x = x_jax
        self.save_from_gc = (x_cupy, x_dlpack, x_cpu)
        self.stream = stream

    def get(self):
        if self.stream is not None:
            self.stream.synchronize()
        else:
            assert self.save_from_gc[0] is None

        return self.x

def to_single_device(x, put_device):
    if isinstance(x, int) or x.dtype == jax.float0:
        return x

    if isinstance(x, np.ndarray) or is_cpu(x):
        return x

    devices = x.devices()
    if len(devices) == 1:
        return x

    # choose a random device
    x = jax.device_put(x, put_device)
    return x

def dlpack_blocking_gpu2cpu(x):
    stream = make_io_stream()
    cpu_x, block_fn = dlpack_gpu2cpu(x, stream)
    x, cpu_x = block_fn()
    return cpu_x

import numpy as np
def dlpack_gpu2cpu(x, stream, replace_buffers=None):
    devices_per_leaf = [] #  jax.tree.leaves(x)[0].devices()
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, 'devices'):
            devices_per_leaf.append(leaf.devices())

    put_device = None
    if devices_per_leaf:
        for devices in devices_per_leaf:
            if len(devices) > 1:
                # choose random device
                put_device = list(devices)[np.random.randint(0, len(devices))]
                break

    if not (put_device is None):
        device_putter = partial(to_single_device, put_device=put_device)
        x = jax.tree_util.tree_map(device_putter, x)

    map_fn = partial(_dlpack_gpu2cpu, stream=stream)

    if replace_buffers is None:
        cpu_x = jax.tree_util.tree_map(map_fn, x)
    else:
        cpu_x = jax.tree_util.tree_map(map_fn, x, replace_buffers)

    def block_fn():
        stream.synchronize()
        return x, cpu_x

    return cpu_x, block_fn
    # return jax.device_put(x, jax.devices('cpu')[0])
