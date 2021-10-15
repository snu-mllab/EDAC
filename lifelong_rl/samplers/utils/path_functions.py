import numpy as np

# CHANGE: adapted from https://github.com/aravindr93/mjrl/blob/v2/mjrl/utils/process_samples.py

def compute_returns(paths, discount):
    for path in paths:
        path['returns'] = discount_sum(path['rewards'], discount)

def compute_advantages(paths, baseline, discount, gae_lambda=None, normalize=False):
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for path in paths:
            path['baseline'] = baseline.predict(path)
            path['advantages'] = path['returns'] - path['baseline']
        if normalize:
            alladv = np.concatenate([path['advantages'] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path['advantages'] = (path['advantages'] - mean_adv) / (std_adv + 1e-8)
    else:
        for path in paths:
            b = path['baseline'] = baseline.predict(path)
            terminated = path['terminals'][-1]
            if b.ndim == 1:
                b1 = np.append(path['baseline'], 0.0 if terminated else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if terminated else b[-1]))
            td_deltas = path['rewards'] + discount * b1[1:] - b1[:-1]
            path['advantages'] = discount_sum(td_deltas, discount * gae_lambda)
        if normalize:
            alladv = np.concatenate([path['advantages'] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path['advantages'] = (path['advantages'] - mean_adv) / (std_adv + 1e-8)

def discount_sum(x, discount, terminal=0.0):
    y = []
    run_sum = terminal
    for t in range(len(x)-1, -1, -1):
        run_sum = x[t] + discount * run_sum
        y.append(run_sum)
    
    return np.array(y[::-1])
