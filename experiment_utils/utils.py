import os
import json
import numpy as np
import torch
import lifelong_rl.torch.pytorch_util as ptu
from math import ceil


def load_exp_data(exp_path):
    exp_data = None
    try:
        params_json = load_json(os.path.join(exp_path, "variant.json"))
        progress_csv_path = os.path.join(exp_path, "progress.csv")
        pkl_paths = [os.path.join(exp_path, 'offline_itr_2000.pt')]
        exp_data = dict(csv=progress_csv_path,
                        json=params_json,
                        pkl=pkl_paths,
                        exp_name=exp_path)
    except IOError as e:
        print(e)
    return exp_data


def load_json(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data


def load_dataset(env, terminate_on_end=False, **kwargs):
    '''
    Return offline dataset: Dictionary (np array)
    '''
    dataset = env.get_dataset(**kwargs)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    init_indices = []
    episode_step = 0
    added = 0
    N = dataset['rewards'].shape[0]
    for i in range(N - 1):
        if episode_step == 0:
            init_indices.append(added)

        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            episode_step = 0
            continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1
        added += 1

        if done_bool or final_timestep:
            episode_step = 0

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'init_idx': init_indices,
    }
