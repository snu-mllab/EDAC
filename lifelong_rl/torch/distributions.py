import numpy as np
import torch
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Normal as TorchNormal
from torch.distributions import Independent as TorchIndependent

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.util.eval_util import create_stats_ordered_dict


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)


class Distribution(TorchDistribution):

    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class Independent(Distribution, TorchIndependent):

    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class TorchDistributionWrapper(Distribution):

    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()


class MultivariateDiagonalNormal(TorchDistributionWrapper):

    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
            # exclude_max_min=True,
        ))
        stats.update(create_stats_ordered_dict(
            'std',
            ptu.get_numpy(self.distribution.stddev),
        ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = TorchNormal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = atanh(value)
            
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            TorchNormal(
                ptu.zeros(self.normal_mean.size()),
                ptu.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
