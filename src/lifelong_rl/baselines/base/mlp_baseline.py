import numpy as np
import torch
import torch.nn as nn

from lifelong_rl.baselines.base.base import Baseline
import lifelong_rl.torch.pytorch_util as ptu


# CHANGE: adapted from https://github.com/aravindr93/mjrl/blob/v2/mjrl/baselines/mlp_baseline.py
class MlpBaseline(nn.Module, Baseline):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation,
            **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Value function network
        self.layer_size = [self.input_size + 4] + hidden_sizes + [self.output_size]
        self.hidden_activation = hidden_activation
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(self.layer_size[i], self.layer_size[i+1])
                                             for i in range(len(self.layer_size) - 1)])
        
    """
    Network forward
    """
    def forward(self, feat_mat):
        h = feat_mat
        for i in range(len(self.fc_layers) - 1):
            h = self.fc_layers[i](h)
            h = self.hidden_activation(h)
        h = self.fc_layers[-1](h)
        return h
    
    """
    Main functions
    """
    def features(self, paths):
        # TODO: supports other environments
        obs = np.concatenate([path['observations'] for path in paths])
        obs = np.clip(obs, -10, 10) / 10.0  
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        N, n = obs.shape
        num_feat = int(n + 4)
        feat_mat = np.ones((N, num_feat))

        # Linear features
        feat_mat[:, :n] = obs
        k = 0
        for i, path in enumerate(paths):
            l = len(path['rewards'])
            al = np.arange(l) / 1000.0
            for j in range(4):
                feat_mat[k:k+l, -4+j] = al ** (j+1)
            k += l
        
        return feat_mat

    def predict(self, path):
        feat_mat = ptu.from_numpy(self.features([path]))
        prediction = ptu.get_numpy(self(feat_mat)).ravel()
        return prediction
