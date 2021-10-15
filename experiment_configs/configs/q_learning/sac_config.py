from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning.sac import SACTrainer
import lifelong_rl.util.pythonplusplus as ppp
import os
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.nn import functional as F


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
        args=None,
):
    """
    Policy construction
    """

    num_qs = variant['trainer_kwargs']['num_qs']
    num_minqs = variant['trainer_kwargs']['num_minqs']
    M = variant['policy_kwargs']['layer_size']
    num_q_layers = variant['policy_kwargs']['num_q_layers']
    num_p_layers = variant['policy_kwargs']['num_p_layers']

    q_activation = variant['policy_kwargs']['q_activation']

    hidden_activation = None
    if q_activation == 'relu':
        hidden_activation = F.relu
    elif q_activation == 'softplus':
        hidden_activation = F.softplus
    elif q_activation == 'silu':
        hidden_activation = F.silu
    elif q_activation == 'leaky':
        hidden_activation = F.leaky_relu
    else:
        raise NotImplementedError

    qfs, target_qfs = ppp.group_init(
        2,
        ParallelizedEnsembleFlattenMLP,
        ensemble_size=num_qs,
        sample_size=num_minqs,
        hidden_sizes=[M] * num_q_layers,
        input_size=obs_dim + action_dim,
        output_size=1,
        layer_norm=None,
        hidden_activation=hidden_activation
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_p_layers,
        layer_norm=None,
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qfs=qfs,
        target_qfs=target_qfs,
        replay_buffer=replay_buffer,
        args=args,
        **variant['trainer_kwargs'],
    )
    """
    Create config dict
    """

    config = dict()
    config.update(
        dict(
            trainer=trainer,
            exploration_policy=policy,
            evaluation_policy=MakeDeterministic(policy),
            exploration_env=expl_env,
            evaluation_env=eval_env,
            replay_buffer=replay_buffer,
            qfs=qfs,
        ))
    config['algorithm_kwargs'] = variant['algorithm_kwargs']

    return config
