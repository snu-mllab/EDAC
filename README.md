# Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble

[![DOI](https://zenodo.org/badge/415660116.svg)](https://zenodo.org/badge/latestdoi/415660116) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/snu-mllab/EDAC/blob/main/LICENSE)


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

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).

2. Install Python packages listed in `requirements.txt` using `pip`. You should specify the versions of `mujoco_py` and `dm_control` in `requirements.txt` depending on the version of MuJoCo engine you have installed as follows:
    - MuJoCo 2.0: `mujoco-py<2.1,>=2.0`, `dm_control==0.0.364896371`
    - MuJoCo 2.1.0: `mujoco-py<2.2,>=2.1`, `dm_control==0.0.403778684`
    - MuJoCo 2.1.1: to be updated

3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl). You should remove lines including `dm_control` in `setup.py`.

Here is an example of how to install all the dependencies on Ubuntu:
  
```bash
conda create -n edac python=3.7
conda activate edac
# Specify versions of mujoco-py and dm_control in requirements.txt
pip install --no-cache-dir -r requirements.txt

cd .
git clone https://github.com/rail-berkeley/d4rl.git

cd d4rl
# Remove lines including 'dm_control' in setup.py
pip install -e .
```

## Reproducing the results

### Gym

To reproduce SAC-N results for MuJoCo Gym, run:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs [N]
```

To reproduce EDAC results for MuJoCo Gym, run:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs [N] --eta [ETA]
```

### Adroit

On Adroit tasks, we apply reward normalization for further training stability. For example, to reproduce the EDAC results for pen-human, run:

```bash
python -m scripts.sac --env_name pen-human-v1 --epoch 200 --num_qs 20 --plr 3e-5 --eta 1000 --reward_mean --reward_std
```

To reproduce the EDAC results for pen-cloned, run:

```bash
python -m scripts.sac --env_name pen-human-v1 --epoch 200 --num_qs 20 --plr 3e-5 --eta 10 --max_q_backup --reward_mean --reward_std
```

## Acknowledgement

This work was supported in part by Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd., Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2020-0-00882, (SW STAR LAB) Development of deployable learning intelligence via self-sustainable and trustworthy machine learning and No. 2019-0-01371, Development of brain-inspired AI with human-like intelligence), and Research Resettlement Fund for the new faculty of Seoul National University. This material is based upon work supported by the Air Force Office of Scientific Research under award number FA2386-20-1-4043.
