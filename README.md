# Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble

This is the code for reproducing the results of the paper "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble" accepted at NeurIPS'2021.

This code builds up from the offical code of [Reset-Free Lifelong Learning with Skill-Space Planning](https://sites.google.com/berkeley.edu/reset-free-lifelong-learning), originally derived from [rlkit](https://github.com/vitchyr/rlkit). 

If you find this repository useful for your research, please cite:

```bash
@inproceedings{
    an2021edac,
    title={Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble},
    author={Gaon An and Seungyong Moon and Jang-Hyun Kim and Hyun Oh Song},
    booktitle={Neural Information Processing Systems},
    year={2021}
}
```

## Requirements

* python (3.7.4)
* pytorch (1.7.1)
* gym
* mujoco-py
* d4rl
* CUDA
* numpy

## Dataset preperation

To be updated

## Reproducing the results

To reproduce SAC-N results in MuJoCo Gym, run:

```bash
python run_scripts/sac.py --env_name [ENVIRONMENT] --num_qs [N]
```

To reproduce EDAC results in MuJoCo Gym, run:

```bash
python run_scripts/sac.py --env_name [ENVIRONMENT] --num_qs [N] --eta [ETA]
```

On Adroit tasks, we apply reward normalization for further training stability. For example, to reproduce the EDAC results for pen-human, run:

```bash
python run_scripts/sac.py --env_name pen-human-v1 --num_qs 20 --plr 3e-5 --eta 1000 --reward_mean --reward_std 1.0
```


## License

MIT License
