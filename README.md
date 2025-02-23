# Play-ICE-Lake-by-using-Q-learning
This repository contains a Q-Learning implementation for solving the `FrozenLake-v1` environment from the `gymnasium` library. The goal of this project is to train an agent to navigate a frozen lake, avoiding holes and reaching the goal.
## Table of Contents
- [Introduction](#introduction)
- [Environment](#environment)
- [Q-Learning Algorithm](#q-learning-algorithm)
- [Dependencies](#dependencies)
- [Results](#Results)


## Introduction

The `FrozenLake-v1` environment is a grid-world problem where the agent must navigate from the start (S) to the goal (G) while avoiding holes (H). The ice is slippery, so the agent may not always move in the intended direction. This project uses the Q-Learning algorithm to train an agent to solve this environment.

## Environment
python 3.8
pycharm 24.3.1.1
The environment consists of a 4x4 grid:

## Q-Learning Algorithm

Q-Learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. The algorithm updates the Q-values using the Bellman equation:

\[ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] \]

Where:
- \( Q(s, a) \): The current Q-value for state \( s \) and action \( a \).
- \( \alpha \): The learning rate.
- \( r \): The reward received after taking action \( a \) in state \( s \).
- \( \gamma \): The discount factor.
- \( \max_{a'} Q(s', a') \): The maximum Q-value for the next state \( s' \).

## dependencies
import numpy as np
import gymnasium as gym
import random
import os
import tqdm

import pickle

## Results
After training the agent for 100 episodes, the evaluation results are as follows:

example： Mean_reward=0.75 +/- 0.43

This indicates that the agent successfully reaches the goal 75% of the time on average.


## Run the training and evaluation script:
lake.py


## Check the results:
The script will print the mean reward and standard deviation after evaluation.

The Q-table will be saved to “qtable_frozenlake.pkl”



