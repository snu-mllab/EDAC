from d4rl import qlearning_dataset
import numpy as np
import os
from experiment_utils.utils import load_dataset
import gym


def load_hdf5(env, replay_buffer, args):
    # filename = os.path.split(env.dataset_url)[-1]
    # h5path = os.path.join(D4RL_DIR, filename)

    refined_dataset = qlearning_dataset(env)

    observations = refined_dataset['observations']
    next_obs = refined_dataset['next_observations']
    actions = refined_dataset['actions']
    rewards = np.expand_dims(np.squeeze(refined_dataset['rewards']), 1)

    normalize_mean = True if args.get('reward_mean') else False

    bias = args.get('reward_add', -1e5)
    add_bias = bias > -1e3

    std_scaler = args.get('reward_std', -1)
    normalize_std = True if std_scaler > 0 else False

    scaler = args.get('reward_scale', -1)
    mult_scale = scaler > 0

    reward_pen = args.get('reward_pen', False)

    print("\nRewards stats before preprocessing")
    print('mean:', rewards.mean())
    print('std:', rewards.std())
    print('max:', rewards.max())
    print('min:', rewards.min())

    if reward_pen:
        rewards = np.where(rewards > 30, rewards-50, rewards)
        rewards = np.where(rewards > 5, rewards-10, rewards)
    else:
        if normalize_mean:
            rewards -= rewards.mean()

        if add_bias:
            rewards += bias

        if normalize_std:
            rewards_mean = rewards.mean()
            rewards = (rewards -
                    rewards_mean) / rewards.std() * std_scaler + rewards_mean

        if mult_scale:
            rewards *= scaler

    print("\nRewards stats after preprocessing")
    print('mean:', rewards.mean())
    print('std:', rewards.std())
    print('max:', rewards.max())
    print('min:', rewards.min())

    terminals = np.expand_dims(np.squeeze(refined_dataset['terminals']), 1)
    dataset_size = observations.shape[0]
    '''
    for i in range(len(observations)):
        replay_buffer.add_sample(
            observations[i],
            actions[i],
            rewards[i],
            terminals[i],
            next_obs[i],
        )
    '''
    replay_buffer._observations = observations
    replay_buffer._next_obs = next_obs
    replay_buffer._actions = actions
    replay_buffer._rewards = rewards
    replay_buffer._terminals = terminals

    replay_buffer._size = dataset_size
    replay_buffer.total_entries = dataset_size
    replay_buffer._top = replay_buffer._size

    # Work for state observations
    obs_dim = observations.shape[-1]
    low = np.array(obs_dim * [replay_buffer._ob_space.low[0]])
    high = np.array(obs_dim * [replay_buffer._ob_space.high[0]])
    replay_buffer._ob_space = gym.spaces.Box(low, high)
    replay_buffer._ob_shape = replay_buffer._ob_space.shape
    replay_buffer._observation_dim = obs_dim

    print(f'\nReplay buffer size : {replay_buffer._size}')
    print(f"obs dim            : ", observations.shape)
    print(f"action dim         : ", actions.shape)
    print(f'# terminals: {replay_buffer._terminals.sum()}')
    print(f'Mean rewards       : {replay_buffer._rewards.mean():.2f}')
    replay_buffer._top = replay_buffer._size

    # print('Number of terminals on: ', replay_buffer._terminals.sum())
