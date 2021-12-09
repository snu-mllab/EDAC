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

The environments used in this paper require MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download). The latest version is 2.1.0.

All Python packages required are listed in `requirements.txt`. To install these packages, run the following commands:  

```bash
conda create -n edac python=3.7
conda activate edac
pip install -r requirements.txt
```

Note: For those who have installed MuJoCo 2.0, an error will occur when trying to install mujoco_py and dm_control. To resolve this, you should specify versions as `mujoco-py<2.1,>=2.0` and `dm_control==0.0.364896371` in `setup.py` and manually install the d4rl package as below:
  
```bash
conda create -n edac python=3.7
conda activate edac
# Specify versions of mujoco-py and dm_control and remove d4rl in setup.py
pip install -r requirements.txt

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

This work was supported in part by Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd., Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2020-0-00882, (SW STAR LAB) Development of deployable learning intelligence via self-sustainable and trustworthy machine learning and No. 2019-0-01371, Development of brain-inspired AI with human-like intelligence), and Research Resettlement Fund for the new faculty of Seoul National University. This material is based upon work supported by the Air Force Office of Scientific Research under award number FA2386-20-1-4043.
