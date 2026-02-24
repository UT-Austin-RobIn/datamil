import jax.numpy as jnp
import optax

def make_sched(lr, num_iters, pct_start, pct_final, min_lr_factor,
               final_min_lr_factor, dtype, anneal_type):
    assert lr is None
    assert pct_final == 1
    if dtype in ['tf32', 'float32']:
        dtype = jnp.float32

    if final_min_lr_factor < 1:
        raise ValueError('final_min_lr_factor must be >= 1')
    if min_lr_factor < 1:
        raise ValueError('min_lr_factor must be >= 1')

    if anneal_type == 'linear':
        fn = optax.linear_onecycle_schedule
        kw = {
            'pct_final': jnp.array(pct_final, dtype=dtype),
        }
    elif anneal_type == 'cosine':
        raise ValueError('cosine is not supported')
        print('>> anneal_Type is cosine')
        fn = optax.cosine_onecycle_schedule
        kw = {}
    else:
        raise ValueError(f'Unknown anneal type {anneal_type}')

    learning_sched = fn(transition_steps=num_iters,
                        peak_value=jnp.array(1.0, dtype=dtype),
                        pct_start=jnp.array(pct_start, dtype=dtype),
                        div_factor=min_lr_factor,
                        final_div_factor=final_min_lr_factor, **kw)

    total_sched = learning_sched
    return total_sched