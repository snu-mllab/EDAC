# Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble

[![DOI](https://zenodo.org/badge/415660116.svg)](https://zenodo.org/badge/latestdoi/415660116)


This is the code for reproducing the results of the paper [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548) accepted at NeurIPS'2021.

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

You can install ```lifelong_rl``` as a Python package by running ```pip install -e .```.

## Reproducing the results

### Gym

To reproduce SAC-N results for MuJoCo Gym, run:

```bash
python scripts/sac.py --env_name [ENVIRONMENT] --num_qs [N]
```

To reproduce EDAC results for MuJoCo Gym, run:

```bash
python scripts/sac.py --env_name [ENVIRONMENT] --num_qs [N] --eta [ETA]
```

### Adroit

On Adroit tasks, we apply reward normalization for further training stability. For example, to reproduce the EDAC results for pen-human, run:

```bash
python scripts/sac.py --env_name pen-human-v1 --epoch 200 --num_qs 20 --plr 3e-5 --eta 1000 --reward_mean --reward_std
```

To reproduce the EDAC results for pen-cloned, run:

```bash
python scripts/sac.py --env_name pen-human-v1 --epoch 200 --num_qs 20 --plr 3e-5 --eta 10 --max_q_backup --reward_mean --reward_std
```

## Acknowledgement

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence).

## License

MIT License
