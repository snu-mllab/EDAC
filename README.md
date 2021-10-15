# Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble

This is the code for reproducing the results of the paper "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble" submitted to NeurIPS'2021. This code builds up from the offical code of [Reset-Free Lifelong Learning with Skill-Space Planning](https://sites.google.com/berkeley.edu/reset-free-lifelong-learning), originally derived from [rlkit](https://github.com/vitchyr/rlkit). 
## Requirements

* python (3.7.4)
* pytorch (1.7.1)
* gym
* mujoco-py
* d4rl
* CUDA
* numpy

## Reproducing the results

To reproduce SAC-N results in MuJoCo Gym, run:

```bash
python src/run_scripts/sac.py --env_name [ENVIRONMENT] --num_qs [N]
```

To reproduce EDAC results in MuJoCo Gym, run:

```bash
python src/run_scripts/sac.py --env_name [ENVIRONMENT] --num_qs [N] --sigma [ETA]
```

On Adroit tasks, we apply reward normalization for further training stability. For example, to reproduce the EDAC results for pen-human, run:

```bash
python src/run_scripts/sac.py --env_name pen-human-v1 --num_qs 20 --plr 3e-5 --sigma 1000 --reward_mean --reward_std 1.0
```
