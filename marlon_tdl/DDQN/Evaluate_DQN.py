import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sumo_env_new_state_space import SumoEnvironment
import logging
from gymnasium.spaces import Discrete
from torch.utils.tensorboard import SummaryWriter


env = SumoEnvironment(use_gui=False, max_steps=3600, results=True,
                      rPath="bo/DQNc10",  cfg_file="./nets/bo/run.sumocfg")


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer


# Get number of actions from gym action space
# print(state)

run_name = "DQN_Real_Network"
writer = SummaryWriter(f"runs/{run_name}")
logging.basicConfig(filename=f'./rewards/{run_name}', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


models = {}

env.reset()

for id in env.info.keys():
    n_observations = env.info[id]["State Space"]
    n_actions = env.info[id]["Action Space"]
    models[id] = {
        "policy_net": DQN(n_observations, n_actions).to(device),
        "target_net": DQN(n_observations, n_actions).to(device),
        "action_space": Discrete(n_actions),
        "observation_space": n_observations
    }

    models[id]["policy_net"].load_state_dict(torch.load(
        f"./models/DQNv5/policy_net_checkpoint_10/agent{id}", map_location=torch.device('cpu')))
    models[id]["target_net"].load_state_dict(torch.load(
        f"./models/DQNv5/target_net_checkpoint_10/agent{id}", map_location=torch.device('cpu')))


steps_done = 0
guesses = 0


def select_action(model, state, eps_threshold):
    global steps_done
    global guesses
    sample = random.random()
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model["policy_net"](state).max(1)[1].view(1, 1)
    else:
        guesses += 1
        return torch.tensor([[model["action_space"].sample()]], device=device, dtype=torch.long)


if torch.cuda.is_available():
    num_episodes = 1
else:
    num_episodes = 1

rewards = []
c = 1
eps = 0.0
for i_episode in range(1, num_episodes + 1):

    # Initialize the environment and get it's state
    state = env.reset()
    states = {}
    for id, obs in zip(env.agentIDs, state):
        states[id] = torch.tensor(
            obs, dtype=torch.float32, device=device).unsqueeze(0)

    eps_reward = 0
    t = 1
    for t in count():
        print(f"Step {t}")
        actions = {}
        step_actions = []

        for id in env.agentIDs:
            action = select_action(models[id], states[id], eps)
            actions[id] = action
            step_actions.append(action.item())
        observations, reward, terminated, truncated = env.step(step_actions)
        done = True in terminated

        next_states = {}

        for i, id in zip(range(len(observations)), env.agentIDs):
            next_states[id] = torch.tensor(
                observations[i], dtype=torch.float32, device=device).unsqueeze(0)

        states = next_states

        if done:
            break
