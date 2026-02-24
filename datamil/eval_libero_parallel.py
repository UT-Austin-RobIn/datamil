import os
import numpy as np
import gym
import cv2
import jax
from tqdm import tqdm
from functools import partial
import robosuite.utils.transform_utils as T
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from octo.model.octo_model import OctoModel
from collections import deque

import dlimp as dl
import glob
import json

'''
Task ID to instruction,
0 put both the alphabet soup and the tomato sauce in the basket
1 put both the cream cheese box and the butter in the basket
2 turn on the stove and put the moka pot on it
3 put the black bowl in the bottom drawer of the cabinet and close it
4 put the white mug on the left plate and put the yellow and white mug on the right plate
5 pick up the book and place it in the back compartment of the caddy
6 put the white mug on the plate and put the chocolate pudding to the right of the plate
7 put both the alphabet soup and the cream cheese box in the basket
8 put both moka pots on the stove
9 put the yellow and white mug in the microwave and close it
'''

name_to_id = {
    'soup-sauce': 0,
    'cream-butter': 1,
    'stove-moka': 2,
    'bowls-cabinet': 3,
    'bowl-cabinet': 3,
    'mug-mug': 4,
    'book-caddy': 5,
    'mug-pudding': 6,
    'soup-cheese': 7,
    'moka-moka': 8,
    'mug-microwave': 9   
}


def write_video(video_path, frames, fps=30):
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def stack_and_pad(history: list, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - max(num_obs, horizon)
    pad_mask = np.ones(horizon)
    pad_mask[:pad_length] = 0
    full_obs["pad_mask"] = pad_mask
    return full_obs

class OctoEvalWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, metadata, window_size=1):
        super().__init__(env)

        self.metadata = metadata
        self.window_size = window_size
        self.history = deque(maxlen=self.window_size)
        self.num_obs = 0

    def process_obs(self, obs):
        processed_obs = {
            'proprio': np.concatenate((obs['robot0_eef_pos'], T.quat2axisangle(obs['robot0_eef_quat']), [0], obs['robot0_gripper_qpos'][:1]), axis=-1).astype(np.float32),
            'image_primary': np.array(dl.transforms.resize_image(np.flip(obs['agentview_image'], axis=0).astype(np.uint8), size=(256, 256))).astype(float),
            'image_wrist': np.array(np.flip(obs['robot0_eye_in_hand_image'], axis=0)).astype(float)
        }

        # normalize proprio
        processed_obs['proprio'] = self.normalize(processed_obs['proprio'], self.metadata['proprio'])

        return processed_obs
    
    def step(self, action):
        # unnormalize action
        action = self.unnormalize(action, self.metadata['action'])
        action[-1] = 1-2*(action[-1])
        # action[-1] = 2*action[-1] - 1 
        
        n_obs, reward, done, info = self.env.step(action)
        n_obs = self.process_obs(n_obs)
        self.num_obs += 1
        self.history.append(n_obs)
        assert len(self.history) == self.window_size
        full_obs = stack_and_pad(self.history, self.num_obs)
        
        return full_obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        self.num_obs = 1
        self.history.extend([self.process_obs(obs)] * self.window_size)

        full_obs = stack_and_pad(self.history, self.num_obs)
        return full_obs

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data - metadata["mean"]) / (metadata["std"] + 1e-8),
            data,
        )
    
    def unnormalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        mask[-1] = False
        return np.where(
            mask,
            (data * metadata["std"]) + metadata["mean"],
            data,
        )
    
    def set_metadata(self, metadata):
        self.metadata = metadata

class Libero10EnvGenerator:
    def __init__(self, num_env=5, wrapper_kwargs={}):
        self.task_suite = benchmark.get_benchmark("libero_10")()
        self.num_env = num_env
        self.wrapper_kwargs = wrapper_kwargs
        
        self.load_all_tasks()


    def load_all_tasks(self):
        self.env_args = []
        self.task_instructions = []
        self.init_states = []
        for task_id in range(10):
            task = self.task_suite.get_task(task_id)
            task_description = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self.env_args.append({
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            })
            self.task_instructions.append(task_description)
            self.init_states.append(self.task_suite.get_task_init_states(task_id))
    
    def get_env(self, task_id):
        env = SubprocVectorEnv(
            [lambda: OctoEvalWrapper(OffScreenRenderEnv(**self.env_args[task_id]), **self.wrapper_kwargs) for _ in range(self.num_env)]
        )
        
        return env
    
    def set_init_state(self, env, init_set, task_id):
        env.reset()
        env.seed(0)

        init_states = self.init_states[task_id]
        indices = np.arange(init_set*self.num_env, (init_set+1)*self.num_env) % init_states.shape[0]
        init_states_ = init_states[indices]
        env.set_init_state(init_states_)

        dummy_action = np.zeros((self.num_env, 7))
        dummy_action[..., -1] = 1
        for _ in range(20):  # simulate the physics without any actions
            init_obs, _, _, _ = env.step(dummy_action)
        
        return init_obs


def stack_obs(obs):
    obs = {k: np.stack([o[k] for o in obs], axis=0) for k in obs[0].keys()}
    return obs

def eval_task(policy_fn, env, obs, task):
    
    num_envs = len(obs)
    done_mask = np.zeros(num_envs, dtype=bool)
    images = []

    obs = stack_obs(obs)
    images.append(obs['image_primary'][:5])
    actions = []
    for i in tqdm(range(960//FUTURE_ACTION_WINDOW)):

        action = np.array(policy_fn(obs, task), dtype=np.float64)
        for j in range(FUTURE_ACTION_WINDOW):
            actions.append(action[:, j])
            obs, reward, _, info = env.step(action[:, j])
            dones = np.array(env.check_success())
            done_mask = np.logical_or(done_mask, dones)
            if np.all(done_mask):
                break

        obs = stack_obs(obs)
        images.append(obs['image_primary'][:5])
        for k in range(5):
            if done_mask[k]:
                # highlight green if success
                images[-1][k] = (images[-1][k] + np.array([0, 255, 0])) / 2

        if (i+1) % 50 == 0:
            print('dones', np.sum(done_mask))
        if np.all(done_mask):
            return np.sum(done_mask), images, actions
    
    print('dones', np.sum(done_mask))
    return np.sum(done_mask), images, actions

@jax.jit
def sample_actions(
    pretrained_model: OctoModel,
    observations,
    tasks,
    rng
):
    # add batch dim to observations
    # observations = jax.tree_map(lambda x: x[None], observations)
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
    )
    # remove batch dim
    return actions

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped

def main(args, env_gen, env):

    total_evals = 50
    num_envs = args.num_envs
    save_vid = False
    assert total_evals % num_envs == 0
    
    ckpt = args.ckpt
    model = OctoModel.load_pretrained(ckpt)
    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        )
    )

    os.makedirs(args.out_dir, exist_ok=True)
    task_id = args.eval_task
    task_successes = 0
    language_instruction = env_gen.task_instructions[task_id]
    task = model.create_tasks(texts=args.num_envs*[language_instruction])

    print('Evaluating task:', language_instruction)
    for init_set in range(total_evals//num_envs):
        init_obs = env_gen.set_init_state(env, init_set, task_id)
        success_counts, video, actions = eval_task(policy_fn, env, init_obs, task)
        task_successes += success_counts

        video = np.array(video, dtype=np.uint8)
        if save_vid:
            for i in range(5):
                video_path = f"{args.out_dir}/{language_instruction}_{init_set}_{i}.mp4"
                write_video(video_path, video[:, i, 0])
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(np.array(actions)[:, i, -1])
                plt.savefig(f"{args.out_dir}/{language_instruction}_{init_set}_{i}.png")    
    
    # print results
    print(env_gen.task_instructions[task_id])
    print(f"{task_successes}/{total_evals} success rate: {task_successes/total_evals}")
    print(f"{task_successes/total_evals}")
    print()
    
    import json
    with open(f"{args.out_dir}/results.json", "w") as f:
        task_successes_dict = {env_gen.task_instructions[args.eval_task]: task_successes/total_evals}
        json.dump(task_successes_dict, f)
    
    return task_successes/total_evals


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--future_action_window", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=50)

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--task_name", type=str, default=None)
    args = parser.parse_args()

    # task = 'soup-sauce'
    # task = 'cream-butter'
    # task = 'stove-moka'
    # task = 'bowl-cabinet'
    # task = 'mug-mug'
    # task = 'book-caddy'
    # task = 'mug-pudding'
    # task = 'soup-cheese'
    # task = 'moka-moka'
    # task = 'mug-microwave'
    if args.task_name is not None:
        task = args.task_name

    WINDOW_SIZE = args.window_size
    FUTURE_ACTION_WINDOW = args.future_action_window

    args.eval_task = name_to_id[task]
    
    print("Window size:", WINDOW_SIZE)
    print("Future action window:", FUTURE_ACTION_WINDOW)

    env_gen = None
    successes = {}

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir
    ckpt_base= args.ckpt
    for seed in range(5):
        ckpt_files = glob.glob(f'{ckpt_base}/seed{seed}_*')
        print(ckpt_files)
        for ckpt_file in ckpt_files:
            args.out_dir = f'{out_dir}/seed{seed}/{os.path.basename(ckpt_file)}'
            args.ckpt = ckpt_file

            if env_gen is None:
                model = OctoModel.load_pretrained(ckpt_file)
                dataset_statistics = model.dataset_statistics
                env_gen = Libero10EnvGenerator(num_env=args.num_envs, wrapper_kwargs={'metadata': dataset_statistics, 'window_size': WINDOW_SIZE})
                env = env_gen.get_env(args.eval_task)
            print(args.ckpt)
            print(args.out_dir)
            print()
            success = main(args,
                 env_gen=env_gen,
                 env=env)
            successes[os.path.basename(ckpt_file)] = success
    for k in successes:
        print(k, successes[k])
    env.close()
