from torch.utils.data import Dataset, DataLoader
import glob, os
import h5py
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
from datamil.models.mw_model_jax import make_mw_model
from datamil.eval_metaworld import eval_metaworld_sim
from datamil.metagradients.core.optimizers.interpolation import interp_from, interp_from_mom
from datamil.metagradients.core.optimizers.adam import make_adam_optimizer
from functools import partial

EPS = 1.0000000000000001e-11

@partial(jax.jit, static_argnames=['train', 'divisor'])
def per_sample_loss_fn(params,
                       batch,
                       model,
                       train=True,
                       divisor=1.0):

    """
    inputs:
        params: trainable parameters
        frozen_params: non-trainable parameters
    """
    assert divisor == 1.0, divisor
    data = batch
    
    policy = model.replace(params=params)
    bound_module = policy.module.bind({"params": params})
    dist = bound_module(data["observation"])
    action_loss = policy.per_sample_loss(dist, data["action"])

    return action_loss / divisor

max_steps = 4000
lr_scheduler_dict = dict(
    name="cosine",
    init_value=0.0001,
    peak_value=0.005,
    warmup_steps=int(max_steps//25),
    decay_steps=max_steps, # - int(max_steps//25),
    end_value=1e-8,
)
optimizer_dict = dict(
    weight_decay=1e-5,
    clip_gradient=1.0,
    grad_accumulation_steps=None, 
)

OPTIMIZER_KWARGS = {
        'lr': lr_scheduler_dict['peak_value'],
        # 'wd': 1e-5,
        'wd': optimizer_dict['weight_decay'],
        'pct_start': lr_scheduler_dict['warmup_steps'] / lr_scheduler_dict['decay_steps'],
        'pct_final': 1,
        'b1': 0.9,
        'b2': 0.99,
        'min_lr_relative': max(lr_scheduler_dict['init_value'], EPS),
        'final_min_lr_relative': max(lr_scheduler_dict['end_value'], EPS),
        'eps': EPS,
        'eps_sqrt': EPS,
        'selective_wd': True,
        'dtype': jax.numpy.float32,
        'factored_lr_wd': False,
        'anneal_type': 'linear',
        'eps_schedule': jax.tree_util.Partial(interp_from, steps=200,
                                              eps0=1e-08, eps_root0=1e-08, space='geometric'),
        'mom_schedule': jax.tree_util.Partial(interp_from_mom, steps=25, mom0=0.85,
                                        mom1=1, space='linear'),
        'per_param_lr': None,
        'reuse_optimizer': False,
    }


def load_dm_mask(path, topk):
    iter_num = len(os.listdir(path))
    all_dms = []
    count_dms = 0

    n_trajs = 8245
    for iter_id in range(iter_num):
        iter_path = os.path.join(path, f'iter_{iter_id}')
        
        try:
            filtered_dm = np.load(os.path.join(iter_path, 'datamodels.npy'))[1000000:1000000+n_trajs]
            all_dms.append(filtered_dm)
            count_dms += 1
        except:
            pass
    all_dms = np.array(all_dms)
    avg_dm = np.mean(all_dms, axis=0, where=all_dms!=0)
    avg_dm = np.nan_to_num(avg_dm)

    # print(count_dms, np.sum(avg_dm!=0), n_trajs)
    # assert n_trajs == np.sum(avg_dm!=0)

    threshold = np.sort(avg_dm, axis=0)[int(topk*n_trajs)]
    selected_dm = avg_dm <= threshold
    # print(threshold, np.sum(selected_dm))

    return selected_dm

class MetaworldDataset(Dataset):

    def __init__(self, data_folder, path_masks=None):
        super().__init__()

        self.hdf5_files = sorted(glob.glob(os.path.join(data_folder, "**", "*.h5"), recursive=True))
        if path_masks is not None:
            filter_files = []
            for i, file in enumerate(self.hdf5_files):
                if path_masks[i]:
                    filter_files.append(file)
            self.hdf5_files = filter_files

        print(len(self.hdf5_files), "files found")
        
        self.actions = []
        self.observations = []
        for file in tqdm.tqdm(self.hdf5_files):
            with h5py.File(file, 'r') as f:
                self.actions.extend(f['actions'][:])
                self.observations.extend(f['observations'][:])

        # Convert to numpy arrays
        self.actions = np.array(self.actions)
        self.observations = np.array(self.observations)
        
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        observation = self.observations[idx]

        return action, observation


def _grads_and_loss_for_batch(batch, statek, per_sample_loss):
    def losser(params):
        losses = per_sample_loss(params, batch)
        return jnp.sum(losses), losses

    grads, losses = jax.grad(losser, has_aux=True)(statek.params)
    return grads, losses

@jax.jit
def apply_grads(statek, grads):
    return statek.apply_grads(grads)

def train_policy(seed, dm_path, data_path, topk=0.1):
    print(f"==> Training with seed {seed}, topk {topk}")
    selected_mask = load_dm_mask(dm_path, topk=topk)

    dataset = MetaworldDataset(data_path, path_masks=selected_mask)
    print("Dataset Info:-")
    print("Action shape:", dataset.actions.shape)
    print("Observation shape:", dataset.observations.shape, end="\n\n")

    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)#16364

    model = make_mw_model(seed=seed, obs_dim=dataset.observations.shape[1], action_dim=dataset.actions.shape[1])

    psl = jax.tree_util.Partial(
        per_sample_loss_fn,
        model=model,
    )

    state = make_adam_optimizer(
        initial_params=model.params,
        train_its=max_steps,
        **OPTIMIZER_KWARGS,
    )

    n_epochs = int(max_steps / len(dataloader))
    for epoch in tqdm.tqdm(range(n_epochs)):
        total_loss = 0
        for batch in dataloader:
            
            # convert from torch tensor to jax (slow but ok for now)
            actions = jnp.array(batch[0].numpy())
            observations = jnp.array(batch[1].numpy())

            batch = {
                "action": actions,
                "observation": observations,
            }
            grads, loss = _grads_and_loss_for_batch(batch, state, psl)
            state = apply_grads(state, grads)

            total_loss += loss.mean()

    # if (epoch+1) % 250 == 0:
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

    policy = model.replace(params=state.params)
    policy_success = eval_metaworld_sim(policy)['prob']
    print("Success Rate:", policy_success, end="\n\n")
    return [policy_success]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/metaworld/prior/")
    parser.add_argument('--dm_path', type=str, default="results/metaworld/pick-place-wall_dm")
    parser.add_argument('--topk', type=float, default=0.1, help="topk fraction of data to select")
    args = parser.parse_args()

    results = []
    for i in range(5):
        results.append(train_policy(seed=i, dm_path=args.dm_path, data_path=args.data_path, topk=args.topk))
    print('results:', results)
    print('mean:', np.mean(results, axis=0))
    print('std:', np.std(results, axis=0))