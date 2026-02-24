from functools import partial
import jax.numpy as jnp
import jax
from typing import NamedTuple
# from optax import tree_utils as otu
import jax.numpy as jnp
from optax._src import numerics
from jax import tree_util as jtu
from flax import struct
import jax
from functools import partial
from .schedules import make_sched
from .adam import safe_zeros_like
from ..utils import make_shardings

from typing import Any, Optional
def tree_zeros_like(
        tree: Any,
        dtype: Optional[jax.typing.DTypeLike] = None,
    ) -> Any:
    """Creates an all-zeros tree with the same structure.

    Args:
        tree: pytree.
        dtype: optional dtype to use for the tree of zeros.

    Returns:
        an all-zeros tree with the same structure as ``tree``.
    """
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)

class SGDState(NamedTuple):
    count: object
    beta: object

@jax.jit
def recover_grads(this_mu, next_mu, momentum):
    def safe_map(x, y):
        if x is None or x.dtype == jax.float0: return x
        if y is None or y.dtype == jax.float0: return y
        return x - y * momentum
    return jax.tree_util.tree_map(safe_map, next_mu, this_mu)

class SGDTrainState(struct.PyTreeNode):
    lr: object = struct.field(pytree_node=False)
    momentum: object = struct.field(pytree_node=True)
    params: object = struct.field(pytree_node=True)
    batch_stats: object = struct.field(pytree_node=True)
    opt_state: object = struct.field(pytree_node=True)
    weight_decay: object = struct.field(pytree_node=True)
    bias_scaler: object = struct.field(pytree_node=True)
    max_lr: object = struct.field(pytree_node=True)

    def zeros_like(self, is_tangent):
        safe_opt_state = safe_zeros_like(self.opt_state, is_tangent)
        safe_params = safe_zeros_like(self.params, is_tangent)
        return self.replace(params=safe_params, opt_state=safe_opt_state,
                            optimizer=self.optimizer)

    def infer_gradient_from(self, next_state):
        momentum = self.momentum
        grads = recover_grads(self.opt_state.beta,
                              next_state.opt_state.beta,
                              momentum)
        return grads

    def apply_grads(self, grads, bs_updates=None):
        # Update params
        step = self.opt_state.count
        lr = self.lr(step)
        mom = self.momentum
        bias_scaler = self.bias_scaler
        count_inc = numerics.safe_int32_increment(self.opt_state.count)
        wd = self.weight_decay / self.max_lr

        def update_bt(path, _param, _grad, _bt):
            if 'Conv_0' in str(path[0]) and 'kernel' in str(path[-1]):
                return _bt
            if 'norm' in '/'.join([str(p) for p in path]).lower():
                _grad = _grad + wd * _param / bias_scaler
                return mom * _bt + _grad
            else:
                _grad = _grad + wd * _param
                return mom * _bt + _grad

        def calc_per_param_update(path, _grad, _bt):
            if 'Conv_0' in str(path[0]) and 'kernel' in str(path[-1]):
                return jnp.zeros_like(_grad)
            return _grad + mom * _bt

        def apply_per_param_update(path, _param, _update):
            if 'Conv_0' in str(path[0]) and 'kernel' in str(path[-1]):
                return _param
            if 'norm' in '/'.join([str(p) for p in path]).lower():
                return _param - _update * lr * bias_scaler
            return _param - _update * lr

        new_bt = jax.tree_util.tree_map_with_path(update_bt, self.params, grads, self.opt_state.beta)
        updates = jax.tree_util.tree_map_with_path(calc_per_param_update, self.params, new_bt)
        new_params = jax.tree_util.tree_map_with_path(apply_per_param_update, self.params, updates)

        if bs_updates is not None:
            return self.replace(params=new_params,
                                opt_state=SGDState(beta=new_bt, count=count_inc),
                                batch_stats=bs_updates)
        return self.replace(params=new_params,
                            opt_state=SGDState(beta=new_bt, count=count_inc))

    @staticmethod
    def create(params, **opt_kwargs):
        state_0 = SGDState(count=jnp.zeros([], jnp.int32),
                           beta=tree_zeros_like(params))
        return SGDTrainState(params=params,
                             opt_state=state_0,
                             **opt_kwargs)
