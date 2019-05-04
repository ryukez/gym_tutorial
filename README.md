# About
RL tutorials for OpenAI Gym(https://gym.openai.com/), using PyTorch(https://pytorch.org/).

# Contents
## CartPole_PolicyGradient
Implementation of Policy-Gradient algorithm for CartPole-v1. 

`python3 train.py [--load] [--env=CartPole-v1] [--path=results/]`

You might also train agent on other environments by changing `--env` argument, where observation_space is 1-dim & action_space is discrete. However, code is validated only for CartPole.

# Requirements
- Python >= 3.6
- Gym >= 0.12.1
- PyTorch >= 1.1.0
- Matplotlib >= 3.0.3
