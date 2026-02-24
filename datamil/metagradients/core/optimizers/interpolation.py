import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

def make_eps_schedule(steps, eps0, eps_root0, space='geometric'):
    return jax.tree_util.Partial(interp_from, steps=steps, eps0=eps0,
                                 eps_root0=eps_root0, space=space)

def make_mom_schedule(steps, mom0, mom1, space='linear'):
    return jax.tree_util.Partial(interp_from_mom, steps=steps, mom0=mom0,
                                 mom1=mom1, space=space)

@partial(jax.jit, static_argnames='space')
def _interp(k, a, b, steps, space='linear'):
    a = jnp.array(a).astype(jnp.float32)
    b = jnp.array(b).astype(jnp.float32)
    steps = jnp.array(steps).astype(jnp.float32)
    k = jnp.array(k).astype(jnp.float32)

    predicate = k < steps
    operands = (k, a, b, steps)
    if space == 'geometric':
        mul_factor = (b / a) ** (1 / steps)
        outp = a * (mul_factor ** k)
        true_fn = lambda k, a, b, steps: outp
        false_fn = lambda k, a, b, steps: b
    else:
        # raise ValueError(f"Unknown space {space}")
        outp = a + (b - a) * k / steps
        true_fn = lambda k, a, b, steps: outp
        false_fn = lambda k, a, b, steps: b

    final = jax.lax.cond(predicate, true_fn, false_fn, *operands)
    return final

def interp_from(k, eps, eps_root, steps, eps0, eps_root0, *, space):
    this_eps = _interp(k, eps0, eps, steps, space=space)
    this_eps_root = _interp(k, eps_root0, eps_root, steps, space=space)

    return {
        'eps': this_eps,
        'eps_root': this_eps_root
    }

def interp_from_mom(k, mom1, steps, mom0, *, space):
    mom = _interp(k, mom0, mom1, steps, space=space)
    return mom