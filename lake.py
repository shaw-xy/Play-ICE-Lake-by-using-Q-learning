import numpy as np
import gymnasium as gym
import random
import os
import tqdm

import pickle
from tqdm import tqdm


# setting enviroment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
desc=["SFFF", "FHFH", "FFFH", "HFFG"]
gym.make('FrozenLake-v1', desc=desc, is_slippery=True)

#get total state（space） and total action
state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")


# create Qtable(state_space, action_space) and initialized each values by using np.zeros
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest q-value related action
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # # else --> exploration
    else:
        action = env.action_space.sample()

    return action

# training function
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration )
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            # reward standard
            if terminated and reward == 1:
                reward += 100  # win
            elif terminated and reward == 0:
                reward -= 10  # go into trap
            else:
                reward -= 0.1  # negative reward at each step

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            if terminated or truncated:
                break
            state = new_state
    return Qtable
#evaluate function
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):

    # Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    # param env: The evaluation environment
    # n_eval_episodes: Number of episode to evaluate the agent
    # Q: The Q-table
    # param seed: The evaluation seed array

    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

# save Q-table to file
def save_q_table(Qtable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(Qtable, f) #Overwrite, instead of appending

# load existing Q-table
def load_q_table(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#set Training parameters
n_training_episodes = 100  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of  episodes

# Environment parameters
env_id = "FrozenLake-v1"  #name of environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob，use it to decay learning rate


# check exist Qtable
if os.path.exists('qtable_frozenlake.pkl'):
    print("exist!!!!!")
    #exist load
    Qtable_frozenlake = load_q_table('qtable_frozenlake.pkl')
else:
    print("not exist!!!!!")
    Qtable_frozenlake = initialize_q_table(state_space, action_space)

#train
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

#save q-table to file
save_q_table(Qtable_frozenlake, 'qtable_frozenlake.pkl')

# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
