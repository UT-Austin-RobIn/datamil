import jax
import os
import gc
USER = os.environ.get('USER', 'nouser')
os.environ['PYTHONHASHSEED'] = '0'
try:
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_" + USER)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
except AttributeError:
    pass

from datamil.metagradients.core.dlpack import dlpack_blocking_gpu2cpu, dlpack_gpu2cpu, make_io_stream
from datamil.metagradients.core.utils import safe_tree_add, add_trees, make_shardings, safe_divide, \
    safe_add, safe_zeros, CastedDict, get_one, add_trees_ignore_none
import os
import gc
import time
from uuid import uuid4
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from datamil.metagradients.core.dataloading import REPLAYMinibatches
jnp.set_printoptions(threshold=1)

class NotANumberError(Exception):
    pass

def make_save_iterations(start_it, train_its, serialize_k):
    save_iterations = list(range(int(start_it), int(train_its), serialize_k)) + [train_its]
    save_iterations = set(save_iterations)
    return save_iterations

def gpu_show():
    if not os.environ.get('SSH_TTY', False):
        print('>> GPU', os.environ['CUDA_VISIBLE_DEVICES'])
        print('>> MACHINE HOSTNAME', os.environ.get('HOSTNAME', 'no host'))
        os.system('nvidia-smi')

def replay_forward(saved_states, train_its, train_batcher, serialize_k,
                   psl_train, start_it):
    sharding, replicated_sharding = make_shardings()
    state = jax.device_put(saved_states[start_it], replicated_sharding)

    iterator = tqdm(range(start_it, train_its))
    save_iterations = make_save_iterations(start_it, train_its, serialize_k)
    save_iterations.add(start_it+1)

    gpu_states = []
    old_cpu_states = None
    old_cpu_blocker = None
    stream = make_io_stream()

    def flush_and_replace(gpu_states, old_cpu_states, old_cpu_blocker):
        if old_cpu_states is not None:
            old_cpu_blocker()

            # save to disk
            for i, cpu_state in old_cpu_states:
                saved_states.set(i, cpu_state, disk=True)

        # now move gpu_states to cpu
        if len(gpu_states) == 0:
            return [], [], lambda: None

        if old_cpu_states is not None:
            old_cpu_state_values = [v[1] for v in old_cpu_states]
            old_cpu_state_values = old_cpu_state_values[:len(gpu_states)]
        else:
            old_cpu_state_values = None

        gpu_state_values = [v[1] for v in gpu_states]
        gpu_state_indices = [v[0] for v in gpu_states]
        cpu_states, block_fn = dlpack_gpu2cpu(gpu_state_values, stream=stream,
                                              replace_buffers=old_cpu_state_values)
        old_cpu_states = list(zip(gpu_state_indices, cpu_states))
        return [], old_cpu_states, block_fn

    # indices_by_batch = []

    batches = async_iterator(train_batcher, start_it, train_its, 'train')
    for it in iterator:
        _, minibatches = next(batches)
        state = functional_step(state, minibatches, psl_train, return_indices=False)
        # indices_by_batch.append(batch_indices)

        curr_it = it + 1
        if curr_it in save_iterations:
            gpu_states.append((curr_it, state))

        if len(gpu_states) >= 2:
            tup = flush_and_replace(gpu_states, old_cpu_states, old_cpu_blocker)
            gpu_states, old_cpu_states, old_cpu_blocker = tup

    gpu_states, old_cpu_states, old_cpu_blocker = flush_and_replace(gpu_states,
                                                                    old_cpu_states,
                                                                    old_cpu_blocker)
    tup = flush_and_replace(gpu_states, old_cpu_states, old_cpu_blocker)
    print(f'>> Done: trained from cache @ {start_it} -> it {train_its}')
    return state

def async_iterator(batcher, s, e, part):
    iterator = batcher(s, e)
    next_batch = next(iterator)
    next_minibatch = next_batch.get_minibatches(part)
    for i in range(s, e):
        batch = next_batch
        minibatches = next_minibatch
        try:
            next_batch = next(iterator)
            next_minibatch = next_batch.get_minibatches(part)
        except StopIteration:
            assert i == e - 1
            next_batch = None

        if next_batch is None:
            assert i == e - 1

        yield batch, minibatches

def replay_vjp(*, state, train_batcher, val_batcher, train_its, val_its,
               psl_train, psl_test, vjp_skele, vjp_head, segment_size=20,
               forward_only=False, return_state=False, aux_datasets=None,
               save_dired='/tmp/', return_2nd_to_last_dstate=False,
               per_state_skele=False):
    '''
    state0: initial state; includes step
    train_batcher: (s, e, sharding) -> Iterator[MiniBatchIterator]
    val_batcher: (s, e, sharding) -> Iterator[MiniBatchIterator]
    per_sample_loss: (params, (idx, (x, y))) -> one loss per batch example
    vjp_skele: batch, losser -> eps, (eps -> new_losser, new_batch)
    vjp_head: final parameters -> scalar
    save_dir: where to save the final deps
    cache_dir: where to save the intermediate states
    eval_every: how often to evaluate the model
    '''
    if not per_state_skele:
        state_to_vjp_skele = lambda state: vjp_skele
    else:
        state_to_vjp_skele = vjp_skele

    assert aux_datasets is not None
    gpu_show()
    debug_mode = bool(os.environ.get('DEBUG', False))

    sharding, replicated_sharding = make_shardings()
    state = jax.device_put(state, replicated_sharding)
    val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)
    train_batcher = jax.tree_util.Partial(train_batcher, sharding=sharding)
    for k, v in aux_datasets.items():
        aux_datasets[k] = (jax.tree_util.Partial(v[0], sharding=sharding), v[1])

    saved_states_path = Path(save_dired) / f'{USER}_replay_states'/ str(uuid4())
    saved_states_path.mkdir(parents=True)
    saved_states = CastedDict(state, True, saved_states_path)
    saved_dstates = CastedDict(state)

    saved_deps = {}

    if len(saved_deps) > 0 and len(saved_states) == 0:
        raise ValueError('Cannot start from deps without states')

    start_it = int(state.opt_state.count)
    if not start_it in saved_states:
        saved_states.set(start_it, state)

    if segment_size is None:
        segment_size = int(train_its**0.5)

    if forward_only:
        segment_size = 1000000000000

    assert segment_size > 0

    # also get the loss
    evaler = partial(eval_model, val_batcher=val_batcher, val_its=val_its,
                     per_sample_loss=psl_test)

    final_state = replay_forward(saved_states, train_its, train_batcher,
                                 segment_size, psl_train, start_it)

    val_loss = evaler(state=final_state, limited=debug_mode)

    # make list of start, end pairs
    save_iterations = make_save_iterations(start_it, train_its, segment_size)
    save_iterations = sorted(list(save_iterations))
    assert save_iterations[0] == start_it
    all_segments = list(zip(save_iterations[:-1], save_iterations[1:]))[::-1]

    state_cotangents, primal = vjp_head(final_state)
    assert not isinstance(state_cotangents, tuple)
    saved_dstates.set(train_its, state_cotangents)

    assert save_iterations[-1] == int(final_state.opt_state.count)
    curr_step = train_its

    aux_losses = {}
    for aux_name, (val_batcher, aux_its) in aux_datasets.items():
        aux_losses[aux_name] = eval_model(state=final_state,
                                          val_batcher=val_batcher,
                                          val_its=aux_its,
                                          per_sample_loss=psl_test,
                                          limited=debug_mode)
    final_return = {
        'val_loss': float(val_loss),
        'primal': float(primal),
    } | {k: float(v) for k, v in aux_losses.items()}

    print('>> All losses:', final_return)
    # final_return['batch_indices'] = batch_indices
    final_state = dlpack_blocking_gpu2cpu(final_state)

    if forward_only:
        if return_state:
            final_return['final_state'] = final_state

        return final_return

    # filter segments to only include those that are "after" curr_step
    segments = [seg for seg in all_segments if seg[1] <= curr_step]

    # let Jf = Jf_{okaz_N} ... Jf_{okaz_1}
    # Jf_{okaz_i} = Jf_{i * k - 1} ... Jf_{(i - 1) * k}
    print('>> Remaining segments', segments)

    state_cotangents_to_save = []
    if return_2nd_to_last_dstate:
        state_cotangents_to_save = [start_it + 1]
    else:
        state_cotangents_to_save = []

    all_saved_state_cotangents = {}

    prev_start = None
    all_batch_indices = {}
    for start, end in tqdm(segments, desc='Okazaki stages'):
        assert end > start
        ret = replay_stage(final_i=end, start_i=start,
                           train_batcher=train_batcher,
                           psl_train=psl_train,
                           state_to_vjp_skele=state_to_vjp_skele,
                           state_cotangents=state_cotangents,
                           saved_states=saved_states,
                           stage_num=f'{end}/{train_its}',
                           state_cotangents_to_save=state_cotangents_to_save)

        state_cotangents, eps_cotangents, stored_state_cotangents, batch_indices = ret
        all_saved_state_cotangents.update(stored_state_cotangents)

        # eps_cotangents: dictionary from i -> vector of eps_i
        mini = 10
        maxi = -10
        values = [values for _, values in eps_cotangents.items()]
        values = jnp.concatenate(values, axis=0)
        mini = min(mini, values.min())
        maxi = max(maxi, values.max())
        avg = jnp.absolute(values).mean()

        print(f'>> {end} -> {start}: {mini=:.6f} {maxi=:.6f} {avg=:.6f}')
        on_cpu = dlpack_blocking_gpu2cpu(eps_cotangents)
        on_cpu = {k:np.array(v) for k, v in on_cpu.items()}

        batch_indices = dlpack_blocking_gpu2cpu(batch_indices)
        batch_indices = {k: np.array(v) for k, v in batch_indices.items()}
        all_batch_indices.update(batch_indices)

        # TODO: if we want to do LR/WD we need to fix this part to be more general
        on_cpu = {k:v for k, v in on_cpu.items()}
        on_cpu_shifted = {(k - start): v for k, v in on_cpu.items()}
        saved_dstates.set(start, state_cotangents)
        if not prev_start is None:
            del saved_dstates[prev_start]

        prev_start = start
        saved_deps[start] = {(k + start): v for k, v in on_cpu_shifted.items()}

        if pytree_isnan(eps_cotangents):
            raise NotANumberError('NAN ERROR EOM')

    assert len(saved_deps) >= len(all_segments)

    final_return['deps'] = saved_deps
    final_return['batch_indices'] = all_batch_indices
    if return_state:
        final_return['final_state'] = final_state

    final_return['dstates'] = all_saved_state_cotangents
    return final_return

def replay_stage(final_i, start_i, train_batcher, psl_train, state_to_vjp_skele,
                 state_cotangents, saved_states, stage_num,
                 state_cotangents_to_save=[]):
    assert final_i > start_i
    assert start_i in saved_states

    print(f'>> Backward stage: {start_i} -> {final_i}')
    sharding, replicated_sharding = make_shardings()
    state = jax.device_put(saved_states[start_i], replicated_sharding)

    MAX_QUEUE_SIZE = 2
    desc = f'|stage={stage_num} | Retraining segment..'

    gpu_states = {}
    old_states_cpu = None
    stream = make_io_stream()

    def flush_to_memory(gpu_states, old_states_cpu):
        if old_states_cpu is not None:
            old_states_cpu, block_fn = old_states_cpu
            block_fn()
            # get curr stream
            for i, cpu_state in old_states_cpu.items():
                # saved_states.force_set(i, cpu_state)
                saved_states.set(i, cpu_state)

        # now move gpu_states to cpu
        if len(gpu_states) == 0:
            return {}, None

        gpu_keys, gpu_values = zip(*gpu_states.items())
        old_states_cpu, block_fn = dlpack_gpu2cpu(gpu_values, stream=stream)
        old_states_cpu = dict(zip(gpu_keys, old_states_cpu))
        return {}, (old_states_cpu, block_fn)

    batches_iterator = async_iterator(train_batcher, start_i, final_i, 'train')
    saved_batches = {}
    # used to be final_i - 1
    for curr_it in tqdm(range(start_i, final_i), desc=desc):
        batch, minibatches = next(batches_iterator)
        saved_batches[curr_it] = batch
        if (curr_it < final_i - 1) and (curr_it + 1) in saved_states:
            # we have already saved the next state
            state = jax.device_put(saved_states[curr_it + 1], replicated_sharding)
        elif curr_it < final_i - 1:
            state = functional_step(state, minibatches, psl_train)
            state_it = curr_it + 1
            gpu_states[state_it] = state

            if len(gpu_states) > MAX_QUEUE_SIZE:
                gpu_states, old_states_cpu = flush_to_memory(gpu_states, old_states_cpu)

    gpu_states, old_states_cpu = flush_to_memory(gpu_states, old_states_cpu)
    gpu_states, old_states_cpu = flush_to_memory(gpu_states, old_states_cpu)

    del gpu_states
    del old_states_cpu

    last_prev_seen_state = int(state.opt_state.count) + 1
    del state

    backward_its = list(reversed(range(start_i, final_i)))
    assert len(backward_its) == final_i - start_i
    print('>> backward_its', backward_its)

    all_eps_cotangents = {}

    state_p1 = jax.device_put(saved_states[final_i], replicated_sharding)
    state = jax.device_put(saved_states[final_i - 1], replicated_sharding)
    batch = saved_batches[final_i - 1]
    minibatches = batch.get_minibatches('meta')

    gc.collect()
    dstates_saved = {}
    all_batch_indices = {}
    for back_it in tqdm(backward_its, desc=f'|stage={stage_num} | Backward'):
        assert back_it < last_prev_seen_state
        if back_it != start_i:
            nbatch = saved_batches[back_it - 1]
            nminibatches = nbatch.get_minibatches('meta')
            nstate = jax.device_put(saved_states[back_it - 1], replicated_sharding)
            if back_it - 1 != start_i:
                del saved_states[back_it - 1]
        else:
            nbatch, nminibatches, nstate = None, None, None

        vjp_skele = state_to_vjp_skele(state)(bs=minibatches.bs)
        ret = backward_step(minibatches, state, state_p1, state_cotangents,
                            vjp_skele, psl_train=psl_train)

        eps_cotangents, state_cotangents, batch_indices = ret

        jax.block_until_ready((eps_cotangents, state_cotangents, batch_indices))
        if back_it in state_cotangents_to_save:
            # assert back_it == int(state.opt_state.count)
            dstates_saved[back_it] = state_cotangents

        all_eps_cotangents[back_it] = eps_cotangents
        all_batch_indices[back_it] = batch_indices
        state, batch, state_p1, minibatches = nstate, nbatch, state, nminibatches

    all_saved_states = list(saved_states.keys())
    for saved_i in all_saved_states:
        if saved_i > start_i:
            del saved_states[saved_i]

    return state_cotangents, all_eps_cotangents, dstates_saved, all_batch_indices

def pytree_isnan(pytree):
    flat, _ = jax.tree_util.tree_flatten(pytree)
    for arr in flat:
        if (arr.dtype != jax.float0) and jnp.any(jnp.isnan(arr)):
            return True

    return False

def count_nz(psl_train, mb):
    data_weights = psl_train.keywords['data_weights']
    ixs = mb[0]
    count = float(len(ixs)) if data_weights is None else (data_weights[ixs] != 0).astype(jnp.float32).sum()
    return count

GLOBAL_DSTATE_ZERO = None
def zero_dstate(dstate):
    global GLOBAL_DSTATE_ZERO
    if GLOBAL_DSTATE_ZERO is None:
        oopt_state = dstate.opt_state
        nopt_state = oopt_state._replace(mu=jax.tree_util.tree_map(safe_zeros, oopt_state.mu),
                                        nu=jax.tree_util.tree_map(safe_zeros, oopt_state.nu))
        oopt = dstate.optimizer
        new_keywords = {k: v for k,v in oopt.keywords.items()}
        new_keywords = jax.tree_util.tree_map(safe_zeros, new_keywords)
        noopt = jax.tree_util.Partial(oopt.func, **new_keywords)
        zdstate = dstate.replace(params=jax.tree_util.tree_map(safe_zeros, dstate.params),
                                        opt_state=nopt_state, optimizer=noopt)
        GLOBAL_DSTATE_ZERO = zdstate

    return GLOBAL_DSTATE_ZERO

def backward_step(batch, state, next_state, dstate, vjp_skele,
                  psl_train):
    vjp_skele = jax.tree_util.Partial(vjp_skele)
    deps, dgrads, dstate = backward_apply(state, next_state, dstate,
                                          vjp_skele,
                                          psl_train)
    minibatched_backward = partial(backward_grad, state=state,
                                   dgrads=dgrads,
                                   vjp_skele=vjp_skele,
                                   per_sample_loss=psl_train)
    minibatched_backward_no_state = partial(backward_grad_no_state, state=state,
                                             dgrads=dgrads,
                                             vjp_skele=vjp_skele,
                                             per_sample_loss=psl_train)
    dstate_with_nones = jax.tree_util.tree_map(lambda x: None, dstate)
    def mb_bck_with_zeros(mb):
        num_nz = count_nz(psl_train, mb)
        if num_nz == 0.0:
            this_deps, = minibatched_backward_no_state(mb)
            res = this_deps, dstate_with_nones # zero_dstate(dstate)
        else:
            res = minibatched_backward(mb)

        return (num_nz,) + res

    ret, batch_indices = minibatch_func(mb_bck_with_zeros, batch, return_indices=True,
                                        agg_fn=add_trees_ignore_none)
    try:
        # count, deps2, dstate2 = ret
        _, deps2, dstate2 = ret
    except:
        import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # PREV HAD THIS
    # deps2, dstate2 = jax.tree_util.tree_map(partial(safe_divide, y=count), (deps2, dstate2))
    # deps2, dstate2 = deps2, d
    # deps = jax.tree_util.tree_map(safe_add, deps, deps2)
    # dstate = jax.tree_util.tree_map(safe_add, dstate, dstate2)
    deps = safe_tree_add(deps, deps2)
    dstate = safe_tree_add(dstate, dstate2)
    return deps, dstate, batch_indices

def make_forwards(vjp_skele, per_sample_loss):
    get_grads, apply_grads = factored_functional_step(use_jit=False)
    eps, get_grads, apply_grads = vjp_skele(per_sample_loss, get_grads,
                                            apply_grads)
    return eps, get_grads, apply_grads

def take_vjp(fn, cotangents, *args):
    _, vjp_fn = jax.vjp(fn, *args)
    return vjp_fn(cotangents)

@jax.jit
def backward_grad_no_state(batch, state, dgrads, vjp_skele, per_sample_loss):
    eps, get_grads, _ = make_forwards(vjp_skele, per_sample_loss)

    def state_to_grads(eps):
        return get_grads(eps=eps, state=state, batch=batch)

    return take_vjp(state_to_grads, dgrads, eps)

@jax.jit
def backward_grad(batch, state, dgrads, vjp_skele, per_sample_loss):
    eps, get_grads, _ = make_forwards(vjp_skele, per_sample_loss)

    def state_to_grads(eps, state):
        return get_grads(eps=eps, state=state, batch=batch)

    return take_vjp(state_to_grads, dgrads, eps, state)

@jax.jit
def backward_apply(state, next_state, dstate, vjp_skele,
                   per_sample_loss):
    eps, _, apply_grads = make_forwards(vjp_skele, per_sample_loss)
    grads = state.infer_gradient_from(next_state)

    def state_to_state(eps, grads, state0):
        return apply_grads(eps=eps, grads=grads, state=state0)

    dstate = state.replace(params=dstate.params, opt_state=dstate.opt_state)
    return take_vjp(state_to_state, dstate, eps, grads, state)

@jax.jit
def coupled_backward(batch, state, next_state, dstate, vjp_skele,
                     per_sample_loss):
    print(">> compiling coupled backward...")
    eps, get_grads, apply_grads = make_forwards(vjp_skele, per_sample_loss)
    grads = state.infer_gradient_from(next_state)

    def take_step(eps, grads, state):
        return apply_grads(eps=eps, grads=grads, state=state)

    def find_gradient(eps, state):
        return get_grads(eps=eps, state=state, batch=batch)

    dstate = state.replace(params=dstate.params, opt_state=dstate.opt_state)
    deps, dgrads, dstate = take_vjp(take_step, dstate, eps, grads, state)
    # lets convert dgrads to state
    deps2, dstate2 = take_vjp(find_gradient, dgrads, eps, state)
    deps = add_trees(deps, deps2)
    dstate = add_trees(dstate, dstate2)
    return deps, dstate

@jax.jit
def do_sum(psl):
    return jnp.sum(psl)

def one_eval_step(params, batch, per_sample_loss, tot, acc_loss):
    def summer(minibatch):
        psl = per_sample_loss(params, minibatch, divisor=1.0)
        return float(len(minibatch[0])), do_sum(psl)

    count, this_loss = minibatch_func(summer, batch)
    acc_loss = add_trees(acc_loss, this_loss)
    tot = tot + count
    return tot, acc_loss

def eval_model(state, val_batcher, val_its, per_sample_loss, limited):
    if limited:
        val_its = min(50, val_its)

    acc_loss = np.array(0.)
    tot = np.array(0)
    params = state.params

    batches_iterator = async_iterator(val_batcher, 0, val_its, 'val')
    for _ in tqdm(range(0, val_its), desc='Evaluating model..'):
        _, minibatches = next(batches_iterator)
        tot, acc_loss = one_eval_step(params, minibatches, per_sample_loss, tot,
                                      acc_loss)

    return float(acc_loss / tot)

def batch_for_state(state, train_batch_maker):
    if isinstance(state, int):
        k = state
    else:
        k = int(state.opt_state.count)

    return train_batch_maker(k)

def _grads_for_batch(batch, statek, per_sample_loss):
    def losser(params):
        losses = per_sample_loss(params, batch)
        return jnp.sum(losses)

    grads = jax.grad(losser)(statek.params)
    return grads

grads_for_batch = jax.jit(_grads_for_batch)

@partial(jax.jit, donate_argnums=(2,))
def func_and_acc(f, accumulate, acc, x):
    res = f(x)
    return accumulate(acc, res)

def minibatch_func(func, minibatches, *, acc=None, sharding=None, agg_fn=None,
                   do_tqdm=False, return_indices=False):
    sharding, replicated_sharding = make_shardings()
    bs = minibatches.bs
    bsi = jnp.arange(bs)

    def format_minibatch(mb, s):
        try:
            indices, (x, y) = mb
        except:
            print(mb)
            import pdb; pdb.set_trace()

        e = s + len(indices)
        mb_bsi = bsi[s:e]
        num_devices = len(replicated_sharding.mesh.device_ids) if hasattr(replicated_sharding, 'mesh') else 1
        this_sharding = sharding if len(indices) % num_devices == 0 else replicated_sharding
        return jax.device_put((indices, (x, y), mb_bsi), this_sharding)

    total_seen = 0
    minibatches = iter(minibatches)
    next_minibatch = format_minibatch(next(minibatches), total_seen)
    if do_tqdm:
        pbar = tqdm(total=bs, desc='Minibatching')

    all_indices = []

    while True:
        if next_minibatch is not None:
            minibatch = next_minibatch
            this_idxs = minibatch[0]
            total_seen += len(this_idxs)
            if return_indices:
                all_indices.append(this_idxs)

        try:
            next_minibatch = format_minibatch(next(minibatches), total_seen)
        except StopIteration:
            # assert total_seen == bs
            next_minibatch = None

        grads = func(minibatch)

        if agg_fn is None:
            try:
                acc = grads if acc is None else add_trees(acc, grads)
            except:
                import ipdb; ipdb.set_trace()
        else:
            acc = grads if acc is None else agg_fn(acc, grads)

        if do_tqdm:
            pbar.update(len(minibatch[0]))

        if next_minibatch is None:
            break

    # assert total_seen == bs
    if return_indices:
        all_indices = jnp.concatenate(all_indices, axis=0)
        return acc, all_indices

    return acc

@jax.jit
def apply_grads(statek, grads, lr_factor, wd_factor):
    return statek.apply_grads(grads)

def factored_functional_step(*, use_jit=True, lr_factor=get_one(),
                             wd_factor=get_one(), return_indices=None):
    if use_jit:
        this_grads_for_batch = grads_for_batch
    else:
        assert not return_indices
        this_grads_for_batch = _grads_for_batch

    def get(state, batch, per_sample_loss):
        batch_to_grads = partial(this_grads_for_batch,
                                 per_sample_loss=per_sample_loss, statek=state)
        if isinstance(batch, REPLAYMinibatches):
            num_in_batch = batch.bs
            if num_in_batch == 0:
                return jax.tree_util.tree_map(safe_zeros, state.params)

        if use_jit:
            zparams = jax.tree_util.tree_map(safe_zeros, state.params)
            def factored_batch_to_grads(batch):
                num_nz = count_nz(per_sample_loss, batch)
                if num_nz == 0.0:
                    ret = zparams
                else:
                    ret = batch_to_grads(batch)

                return num_nz, ret

            # if return_indices:
            #     num_samples, grads_acc = minibatch_func(factored_batch_to_grads,
            #                                             batch,
            #                                             return_indices=True)
            # else:
            ret = minibatch_func(factored_batch_to_grads, batch,
                                 return_indices=return_indices)
            if return_indices:
                (_, grads_acc), indices = ret
            else:
                _, grads_acc = ret

            # if _ == 0:
            #     raise ValueError('No samples in batch')
            # grads_acc = jax.tree_util.tree_map(partial(safe_divide, y=batchl_+),
            #                          grads_acc)
        else:
            grads_acc = batch_to_grads(batch)

        if return_indices:
            return grads_acc, indices

        return grads_acc

    def apply(state, grads_acc, lr_factor=lr_factor, wd_factor=wd_factor):
        return apply_grads(state, grads_acc, lr_factor, wd_factor)

    return get, apply

def functional_step(state, batch, per_sample_loss, lr_factor=get_one(),
                    wd_factor=get_one(), *, use_jit=True, return_indices=False):
    fns = factored_functional_step(use_jit=use_jit, lr_factor=lr_factor,
                                   wd_factor=wd_factor,
                                   return_indices=return_indices)
    get_grads, apply_grads = fns
    if return_indices:
        grads, batch_indices = get_grads(state, batch, per_sample_loss)
    else:
        grads = get_grads(state, batch, per_sample_loss)

    ret = apply_grads(state, grads, lr_factor, wd_factor)
    if return_indices:
        return ret, batch_indices

    return ret
