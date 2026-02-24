import jax
from functools import partial
import jax.numpy as jnp
from datamil.metagradients.core.vjp import minibatch_func, async_iterator, replay_vjp
from datamil.metagradients.core.utils import make_shardings
from tqdm import tqdm

@jax.jit
def one_sample_vjp(test_batch, state, per_sample_loss):
    def losser(state):
        losses = per_sample_loss(state.params, test_batch, divisor=1.0)
        assert losses.shape == (1,)
        return jnp.sum(losses)

    primal, vjper = jax.vjp(losser, state)
    g, = vjper(jnp.ones_like(primal))
    return g, primal

def one_sample_vjp_head(state, *, per_sample_loss, val_batcher, val_its,
                        test_index):
    sel = None
    sharding, replicated_sharding = make_shardings()
    val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)
    batches = async_iterator(val_batcher, 0, val_its, 'meta')
    for it in tqdm(range(val_its)):
        _, minibatches = next(batches)
        for idx, (x, y) in minibatches:
            if (sel := (idx == test_index)).any():
                break

    assert sel is not None, 'test index not found in val batches'
    test_batch = idx[sel], (x[sel], y[sel])

    return one_sample_vjp(test_batch, state, per_sample_loss)

@jax.jit
def minibatch_to_loss_gradient(minibatch, state, per_sample_loss):
    def losser(state):
        params = state.params
        losses = per_sample_loss(params, minibatch, divisor=1.0)
        return jnp.sum(losses)

    primal, jvper = jax.vjp(losser, state)
    g, = jvper(jnp.ones_like(primal))
    return (g, primal)

@jax.jit
def safe_div(x, y):
    if x.dtype == jax.float0:
        return x

    if y.dtype == jax.float0:
        return y

    if x is None or y is None:
        return None

    return x / y

def sample_loss_vjp_head(state, *, per_sample_loss, val_batcher, val_its):
    func = jax.tree_util.Partial(minibatch_to_loss_gradient, state=state,
                                 per_sample_loss=per_sample_loss)

    iterator = async_iterator(val_batcher, 0, val_its, 'meta')

    n = 0
    acc = None
    for _ in range(val_its):
        batch, minibatches = next(iterator)
        acc = minibatch_func(func, minibatches, acc=acc)
        n += batch.bs

    g, primal = jax.tree_util.tree_map(partial(safe_div, y=n), acc)
    print('>> VAL N IN PRIMAL:', n)
    assert len(acc) == 2
    return g, primal

# def dstate_vjp_skeleton(losser0, get_grads, apply_grads, *, bs):
#     eps = jnp.zeros((bs,))
#     def new_get_grads(eps, state, batch):
#         def new_losser(*args, **kwargs):
#             return losser0(*args, **kwargs)

#         return get_grads(state, batch, new_losser)

#     return eps, new_get_grads, apply_grads

def dstate_vjp_skeleton(losser0, get_grads, apply_grads, *, bs):
    data_weights = losser0.keywords['data_weights']
    eps = jnp.zeros(bs)
    def new_get_grads(eps, state, batch):
        def new_losser(*args, **kwargs):
            # ixs, _, bsi = batch
            # this_eps = eps[bsi]
            # curr_keywords = losser0.keywords
            # adjusted_weights = data_weights.at[ixs].add(this_eps)
            # new_keywords = {k: v for k,v in curr_keywords.items()}
            # new_keywords['data_weights'] = adjusted_weights
            # new_losser0 = jax.tree_util.Partial(losser0.func, **new_keywords)
            # return new_losser0(*args, **kwargs)
            return losser0(*args, **kwargs)

        return get_grads(state, batch, new_losser)

    def new_apply_grads(eps, state, grads):
        return apply_grads(state, grads)

    return eps, new_get_grads, new_apply_grads

# SKELETONS
def example_loss_vjp_skeleton(losser0, get_grads, apply_grads, *, bs):
    data_weights = losser0.keywords['data_weights']
    eps = jnp.zeros(bs)
    def new_get_grads(eps, state, batch):
        def new_losser(*args, **kwargs):
            ixs, _, bsi = batch
            this_eps = eps[bsi]
            curr_keywords = losser0.keywords
            adjusted_weights = data_weights.at[ixs].add(this_eps)
            new_keywords = {k: v for k,v in curr_keywords.items()}
            new_keywords['data_weights'] = adjusted_weights
            new_losser0 = jax.tree_util.Partial(losser0.func, **new_keywords)
            return new_losser0(*args, **kwargs)

        return get_grads(state, batch, new_losser)

    def new_apply_grads(eps, state, grads):
        return apply_grads(state, grads)

    return eps, new_get_grads, new_apply_grads
