from datamil.metagradients.core.optimizers.adam import make_adam_optimizer
from datamil.metagradients.core.utils import set_dtype, make_shardings
import jax.numpy as jnp
import jax
from functools import partial
from operator import getitem
from datamil.metagradients.core.vjp import replay_vjp

def clean_jaxdict(d):
    to_write = {k: str(v) for k,v in d.items()}
    return to_write

@partial(jnp.vectorize, signature='(n),()->()')
def cross_entropy_loss(logits, label):
    def actual_loss(logits, label):
        logits = jax.nn.log_softmax(logits)
        loss = -getitem(logits, label)
        return loss

    def no_loss(logits, label):
        return jnp.zeros((), dtype=jnp.float32)

    return jax.lax.cond(label == -100, no_loss, actual_loss, logits, label)

def vjp_robodm(
        *,
        state,
        vjp_head,
        vjp_skele,
        data_weights,
        return_kw,
        train_batcher,
        val_batcher,
        psl,
        n_train_ba,
        n_val_ba,
        aux_datasets,
        forward_only,
        segment_size=20,
        return_state=False
    ):
    set_dtype('tf32', True)
    # curr_kw = inspect.currentframe().f_locals
    # curr_kw = clean_jaxdict(curr_kw)
    # del curr_kw['state']
    # print('>> Running okazaki_lm with:', curr_kw)

    sharding, replicated_sharding = make_shardings()

    data_weights = jax.device_put(data_weights, replicated_sharding)

    # this is not the thing we need
    # psl = jax.tree_util.Partial(lm_per_sample_loss, model=model)

    # these are the things we need
    psl_test = jax.tree_util.Partial(partial(psl, train=False), data_weights=None)
    psl_train = jax.tree_util.Partial(partial(psl, train=True), data_weights=data_weights)

    psl_head = jax.tree_util.Partial(partial(psl, train=False))

    sharding, replicated_sharding = make_shardings()
    val_batcher_head = jax.tree_util.Partial(val_batcher, sharding=sharding)

    vjp_head = jax.tree_util.Partial(vjp_head, per_sample_loss=psl_head,
                                     val_batcher=val_batcher_head,
                                     val_its=n_val_ba)
    vjp_skele = jax.tree_util.Partial(vjp_skele)

    kw = dict(state=state,
              train_batcher=train_batcher,
              val_batcher=val_batcher,
              train_its=n_train_ba,
              val_its=n_val_ba,
              psl_train=psl_train,
              psl_test=psl_test,
              vjp_skele=vjp_skele,
              vjp_head=vjp_head,
              segment_size=segment_size,
              aux_datasets=aux_datasets,
              forward_only=forward_only,
              return_state=return_state)

    if return_kw:
        return kw

    return replay_vjp(**kw)
