import numpy as np
import torch
from torch import nn as nn

from lifelong_rl.policies.base.base import ExplorationPolicy
from lifelong_rl.torch.pytorch_util import eval_np
from torch.distributions import Normal
import lifelong_rl.torch.pytorch_util as ptu


# CHANGE: adapted from https://github.com/aravindr93/mjrl/blob/v2/mjrl/policies/gaussian_mlp.py
class GaussianPolicy(nn.Module, ExplorationPolicy):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=torch.tanh,
            min_log_std=-3.0,
            init_log_std=0.0,
            **kwargs
    ):

        super(GaussianPolicy, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_log_std = ptu.ones(self.action_dim) * min_log_std

        # Policy network
        self.layer_size = [self.obs_dim] + hidden_sizes + [self.action_dim]
        self.hidden_activation = hidden_activation
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(self.layer_size[i], self.layer_size[i+1])
                                             for i in range(len(self.layer_size) - 1)])
    
        # Reinitialize the last layer
        for param in list(self.parameters())[-2:]:
            param.data = 1e-2 * param.data
        
        self.log_std = torch.nn.Parameter(ptu.ones(self.action_dim) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.parameters())

        # Transform variables
        self.in_shift, self.in_scale = ptu.zeros(self.obs_dim), ptu.ones(self.obs_dim)
        self.out_shift, self.out_scale = ptu.zeros(self.action_dim), ptu.ones(self.action_dim)

        # Easy access variables
        self.log_std_val = ptu.get_numpy(self.log_std).ravel()
        self.param_shapes = [ptu.get_numpy(p).shape for p in self.trainable_params]
        self.param_sizes = [ptu.get_numpy(p).size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)

    """
    Network forward
    """
    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):  
        # Forward
        h = (obs - self.in_shift) / (self.in_scale + 1e-6)
        for i in range(len(self.fc_layers) - 1):
            h = self.fc_layers[i](h)
            h = self.hidden_activation(h)
        mean = self.fc_layers[-1](h) * self.out_scale + self.out_shift
        log_std = self.log_std
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        # Sampling
        if deterministic:
            action = mean
        else:
            if reparameterize is True:
                action = normal.rsample()
            else:
                action = normal.sample()

        # Compute log likelihood
        log_prob = None
        if return_log_prob:
            log_prob = normal.log_prob(action)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return (action, mean, log_std, log_prob)

    """
    Utility functions
    """
    def get_param_values(self):
        params = torch.cat([p.contiguous().view(-1) for p in self.parameters()])
        return params.clone()

    def set_param_values(self, new_params):
        current_idx = 0
        for idx, param in enumerate(self.parameters()):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            vals = torch.max(vals, self.min_log_std) if idx == 0 else vals
            param.data = vals.clone()
            current_idx += self.param_sizes[idx]
        
        self.log_std_val = ptu.get_numpy(self.log_std).ravel()
        self.trainable_params = list(self.parameters())

    def set_transformations(self, in_shift=None, in_scale=None,
                           out_shift=None, out_scale=None, *args, **kwargs):
        in_shift = self.in_shift if in_shift is None else ptu.tensor(in_shift)
        in_scale = self.in_scale if in_scale is None else ptu.tensor(in_scale)
        out_shift = self.out_shift if out_shift is None else ptu.tensor(out_shift)
        out_scale = self.out_scale if out_scale is None else ptu.tensor(out_scale)
        self.in_shift, self.in_scale = in_shift, in_scale
        self.out_shift, self.out_scale = out_shift, out_scale

    """
    Main functions
    """
    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def mean_LL(self, obs, actions):
        _, mean, log_std, _ = self(obs, deterministic=True)
        zs = (actions - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.action_dim * np.log(2 * np.pi)
        return mean, LL
    
    def log_likelihood(self, obs, actions):
        mean, LL = self.mean_LL(obs, actions)
        return ptu.get_numpy(LL)

    def mean_kl(self, obs):
        _, new_mean, new_log_std, *_ = self(obs, deterministic=True)
        old_mean, old_log_std = new_mean.detach(), new_log_std.detach()
        return self.kl_divergence(new_mean, old_mean, new_log_std, old_log_std)

    def kl_divergence(self, new_mean, old_mean, new_log_std, old_log_std):
        new_std, old_std = torch.exp(new_log_std), torch.exp(old_log_std)
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)
