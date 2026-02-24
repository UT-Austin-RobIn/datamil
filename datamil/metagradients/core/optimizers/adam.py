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
from ..utils import make_shardings

CACHED_PARTIAL_CONSTRUCTOR = None

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

class AdamState(NamedTuple):
    count: object
    mu: object
    nu: object

def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jtu.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)

def update_moment_per_elem_norm(updates, moments, decay, order, eps=None):
    """Compute the EMA of the `order`-th moment of the element-wise norm."""
    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g ** order
        else:
            half_order = order / 2
        # JAX generates different HLO for int and float `order`
        if half_order.is_integer():
            half_order = int(half_order)

        return numerics.abs_sq(g)

    if eps is None:
        return jtu.tree_map(
            lambda g, t: (1 - decay) * (orderth_norm(g)) + decay * t, updates, moments)
    else:
        return jtu.tree_map(
            lambda g, t: (1 - decay) * (orderth_norm(g) + eps) + decay * t, updates, moments)

@partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree_util.tree_map(
      lambda t: t / bias_correction_.astype(t.dtype), moment)

def tree_div(tree_x, factor):
    def safe_div(y, x):
        y_is_float = jnp.issubdtype(y.dtype, jnp.floating) and \
            not y.dtype == jax.float0

        if y_is_float:
            return y / x

        return y

    return jax.tree_util.tree_map(safe_div, tree_x, factor)

def safe_zeros_like(x, make_tangent):
    def mapper(x):
        if x is None:
            return None

        if x.dtype == jax.float0:
            return x

        if make_tangent:
            if x.dtype in [jnp.int32, jnp.int64]:
                return None

        return jnp.zeros_like(x)

    return jtu.tree_map(mapper, x)

def make_adam_optimizer(initial_params, train_its, lr, wd, pct_start, pct_final,
                        b1, b2, min_lr_relative, final_min_lr_relative, eps,
                        eps_sqrt, selective_wd, dtype, factored_lr_wd=False,
                        anneal_type=None, eps_schedule=None, mom_schedule=None,
                        per_param_lr=-1, reuse_optimizer=None):
    ps = locals()
    # remove initial_params
    del ps['initial_params']
    print('>> OPTIMIZER PARAMS', ps)

    assert reuse_optimizer is not None
    global CACHED_PARTIAL_CONSTRUCTOR
    sharding, replicated_sharding = make_shardings()

    base_optimizer = AdamOptimizer

    def construct_unjittable_opt_state():
        assert per_param_lr != -1
        if dtype in ['tf32', 'float32']:
            this_dtype = jnp.float32
        else:
            this_dtype = dtype

        min_lr_factor = jnp.array(1/min_lr_relative, dtype=this_dtype)
        final_min_lr_factor = jnp.array(1/final_min_lr_relative, dtype=this_dtype)
        min_lr_factor = jax.device_put(min_lr_factor, replicated_sharding)
        final_min_lr_factor = jax.device_put(final_min_lr_factor, replicated_sharding)

        assert min_lr_factor >= 1
        lr_sched = make_sched(None, train_its, pct_start, pct_final,
                              min_lr_factor, final_min_lr_factor, this_dtype,
                              anneal_type)
        nondiff_opt_state = UnjitADAMOptState(selective_wd=selective_wd,
                                              factored_lr_wd=factored_lr_wd,
                                              lr=lr_sched, eps_schedule=eps_schedule,
                                              mom_schedule=mom_schedule,
                                              per_param_lr=per_param_lr,
                                              linked_momentum=True)
        return nondiff_opt_state

    if reuse_optimizer and CACHED_PARTIAL_CONSTRUCTOR is not None:
        print('>> REUSING OPTIMIZER')
        partial_constructor = CACHED_PARTIAL_CONSTRUCTOR
    else:
        nondiff_opt_state = construct_unjittable_opt_state()
        partial_constructor = partial(base_optimizer,
                                      unjittable_opt_state=nondiff_opt_state)
        if reuse_optimizer:
            print('>> SAVING OPTIMIZER FOR LATER')
            CACHED_PARTIAL_CONSTRUCTOR = partial_constructor

    # now make the diff opt state
    diff_opt_state = JittableADAMOptState(wd=wd, b1=b1, b2=b2, eps=eps,
                                          eps_root=eps_sqrt, max_lr=lr)
    optimizer_maker = jax.tree_util.Partial(partial_constructor,
                                            jittable_opt_state=diff_opt_state)
    return optimizer_from_states(initial_params, optimizer_maker)

def optimizer_from_states(initial_params, optimizer_maker):
    state = TrainState.create(optimizer_maker, initial_params)
    return state

@jax.jit
def recover_grads(this_mu, next_mu, b1):
    def safe_map(x, y):
        if x is None or x.dtype == jax.float0:
            return x

        if y is None or y.dtype == jax.float0:
            return y

        return (x - y * b1)/(1 - b1)

    return jax.tree_util.tree_map(safe_map, next_mu, this_mu)

class TrainState(struct.PyTreeNode):
    params: object = struct.field(pytree_node=True)
    opt_state: object = struct.field(pytree_node=True)
    optimizer: object = struct.field(pytree_node=True)

    def zeros_like(self, is_tangent):
        safe_opt_state = safe_zeros_like(self.opt_state, is_tangent)
        safe_params = safe_zeros_like(self.params, is_tangent)
        return self.replace(params=safe_params, opt_state=safe_opt_state,
                            optimizer=self.optimizer)

    def apply_grads_auto(self, state, grads):
        assert isinstance(state, TrainState)
        res = self.optimizer().auto_step(state, grads)
        assert isinstance(res, TrainState)
        assert res.optimizer == self.optimizer
        assert not res.opt_state.count is None
        return res

    def trim(self):
        return self.opt_state.mu

    def infer_gradient_from(self, next_state):
        this_mu = self.opt_state.mu
        if hasattr(next_state, 'opt_state'):
            next_mu = next_state.opt_state.mu
        else:
            next_mu = next_state

        optimizer = self.optimizer()
        linked_momentum = optimizer.linked_momentum
        b1 = get_momentum(optimizer, self.opt_state.count, optimizer.b1,
                          optimizer.b2, linked_momentum)[0]
        grads = recover_grads(this_mu, next_mu, b1)
        return grads

    def jvp_auto(self, *, grads, dgrads, dstate):
        step = self.apply_grads_auto

        if dstate is None:
            step = partial(step, self)
            out, dout = jax.jvp(step, (grads,), (dgrads,))
        else:
            out, dout = jax.jvp(step, (self, grads), (dstate, dgrads))

        # assert that jvp is the same
        assert isinstance(dout, TrainState)
        return out, dout

    def apply_grads(self, grads):
        return self.apply_grads_auto(self, grads)

    def jvp(self, grads, dgrads, dstate):
        res = self.jvp_auto(grads=grads, dgrads=dgrads, dstate=dstate)
        primal, tangent = res
        assert primal.opt_state.count is not None
        tangent_count = tangent.opt_state.count
        assert tangent_count is None or tangent_count.dtype == jax.float0
        return primal, tangent

    @staticmethod
    def create(optimizer, params):
        initialized_opt = optimizer()
        state_0 = initialized_opt.initial_state(params)
        return TrainState(params=params, opt_state=state_0,
                          optimizer=optimizer)

def constant_eps(k, eps, eps_root):
    return {
        'eps': eps,
        'eps_root': eps_root
    }

class JittableADAMOptState(NamedTuple):
    wd: float
    b1: float
    b2: float
    eps: float
    eps_root: float
    max_lr: float

class UnjitADAMOptState(NamedTuple):
    selective_wd: bool
    factored_lr_wd: bool
    lr: object # function
    eps_schedule: object # function
    mom_schedule: object # function
    per_param_lr: object # function
    linked_momentum: bool

class AdamOptimizer:
    def __init__(self, jittable_opt_state, unjittable_opt_state):
        selective_wd = unjittable_opt_state.selective_wd
        factored_lr_wd = unjittable_opt_state.factored_lr_wd
        lr = unjittable_opt_state.lr
        eps_schedule = unjittable_opt_state.eps_schedule
        mom_schedule = unjittable_opt_state.mom_schedule
        per_param_lr = unjittable_opt_state.per_param_lr

        wd = jittable_opt_state.wd
        b1 = jittable_opt_state.b1
        b2 = jittable_opt_state.b2
        eps = jittable_opt_state.eps
        eps_root = jittable_opt_state.eps_root
        max_lr = jittable_opt_state.max_lr

        if eps_schedule is None:
            print('>> 🚨 Using constant eps')
            eps_schedule = constant_eps

        self.per_param_lr = per_param_lr
        assert per_param_lr != -1
        self.epser = partial(eps_schedule, eps=eps, eps_root=eps_root)
        self.lr = lr
        self.linked_momentum = unjittable_opt_state.linked_momentum
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.selective_wd = selective_wd
        self.mom_schedule = mom_schedule
        self.factored_lr_wd = factored_lr_wd
        self.max_lr = max_lr
        assert callable(lr)

    def get_momentum_factor(self, k):
        return self.mom_schedule(k)

    def eps(self, k):
        return self.epser(k)['eps']

    def eps_root(self, k):
        return self.epser(k)['eps_root']

    def initial_state(self, params):
        mu = tree_zeros_like(params)
        nu = tree_zeros_like(params)
        _, replicated_sharding = make_shardings()
        count = jax.device_put(jnp.zeros([], jnp.int32), replicated_sharding)
        return AdamState(count=count, mu=mu, nu=nu)

    def auto_step(self, state, updates):
        ret = old_adam_step(self, state.params, state.opt_state, updates,
                             AdamState, per_param_lr=self.per_param_lr)
        new_params, new_adam_state = ret
        return state.replace(params=new_params, opt_state=new_adam_state)

def old_adam_step(opt, params, state, updates, AdamState, per_param_lr):
    selective_wd = opt.selective_wd
    linked_momentum = opt.linked_momentum
    b1, b2 = get_momentum(opt, state.count, opt.b1, opt.b2, linked_momentum)
    mu = update_moment(updates, state.mu, b1, 1)
    count_inc = numerics.safe_int32_increment(state.count)
    count = state.count
    eps_root = opt.eps_root(count)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(
        lambda m, v: m * (jax.lax.rsqrt(v + eps_root) + eps_root), mu_hat, nu_hat)

    # then get the current learning rate
    max_lr = opt.max_lr
    base_lr = opt.lr(count) * max_lr
    wd = opt.wd

    def per_param_update(path, p, u):
        if per_param_lr is not None:
            this_lr_fac = per_param_lr(path)
        else:
            this_lr_fac = 1.

        lr = base_lr * this_lr_fac
        wd_strength = wd * lr if not opt.factored_lr_wd else wd * (base_lr / max_lr)
        not_ln_or_embed = p.ndim > 1 and ('embed' not in str(path))
        if not_ln_or_embed or not selective_wd:
            # if no selective wd, or if ndim > 1
            p = p * (1. - wd_strength)

        param_delta = u * lr
        return p - param_delta

    new_params = jtu.tree_map_with_path(per_param_update, params, updates)
    return new_params, AdamState(count=count_inc, mu=mu, nu=nu)

def get_momentum(opt, count, b1, b2, linked_momentum):
    factor = opt.get_momentum_factor(count)
    this_b1 = b1 * factor
    if linked_momentum:
        separation = (1 - b1)/(1 - b2)
        this_b2 = 1 - (1 - this_b1) / separation
    else:
        this_b2 = b2 * factor

    return this_b1, this_b2
