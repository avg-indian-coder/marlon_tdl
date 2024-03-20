import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gautham_env import SumoEnvironment
from generator import TrafficGen
import logging
from gymnasium.spaces import Discrete
from torch.utils.tensorboard import SummaryWriter

network = "3x3"

if network == "bo":
    from adjacent_nodes import adjacent_nodes_bo
    adj_nodes = adjacent_nodes_bo
elif network == "3x3":
    from adjacent_nodes import adjacent_nodes_3x3
    adj_nodes = adjacent_nodes_3x3


env = SumoEnvironment(use_gui=False, 
                      max_steps=3600,
                      network=network,
                      neighbours=adj_nodes,
                      degree_of_multiagency=1,
                      cfg_file=f"./nets/{network}/run.sumocfg")

run = env.get_run()
with open(f"./DDQN/runs/{network}/run_{run}/logs.csv", "w") as f:
    print(f"episode,rewards,avg_acc_waiting_time,max_acc_waiting_time,epsilon", file=f)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

traffic_generator = TrafficGen(
    # "nets/3x3/3x3.net.xml",
    # "nets/3x3/generated_route.rou.xml",
    f"nets/{network}/network.net.xml",
    f"nets/{network}/generated_route.rou.xml",
    3600,
    2000,
    0.1)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # self.bn = nn.BatchNorm1d(num_features=n_observations, affine=False)
        self.layer1 = nn.Linear(n_observations, n_observations*3)
        self.layer2 = nn.Linear(n_observations*3, n_observations*3)
        self.layer3 = nn.Linear(n_observations*3, n_actions)

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
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
# print(state)

# run_name = "DQN_bo"
# writer = SummaryWriter(f"./DDQN/runs/{run_name}")
# logging.basicConfig(filename=f'./DDQN/rewards/{run_name}', filemode='w',
#                     format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


models = {}

state = env.reset(traffic_generator.generate_routefile, 0)

for id in env.info.keys():
    n_observations = env.info[id]["State Space"]
    n_actions = env.info[id]["Action Space"]
    models[id] = {
        "policy_net": DQN(n_observations, n_actions).to(device),
        "target_net": DQN(n_observations, n_actions).to(device),
        "memory": ReplayMemory(10000),
        "action_space": Discrete(n_actions),
        "observation_space": n_observations
    }

    models[id]["optimizer"] = optim.AdamW(
        models[id]["policy_net"].parameters(), lr=LR, amsgrad=True),
    models[id]["target_net"].load_state_dict(
        models[id]["policy_net"].state_dict())


steps_done = 0
guesses = 0


def select_action(model, state, eps_threshold):
    global steps_done
    global guesses
    sample = random.random()
    steps_done += 1
    # print(policy_net(state))

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model["policy_net"](state).max(1)[1].view(1, 1)
    else:
        guesses += 1
        return torch.tensor([[model["action_space"].sample()]], device=device, dtype=torch.long)


episode_durations = []


# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())
def optimize_model(model, no_of_epochs):

    memory = model["memory"]
    policy_net = model["policy_net"]
    target_net = model["target_net"]
    optimizer = model["optimizer"][0]
    
    policy_net.train(True)
    # print(optimizer)
    # print(len(memory))
    if len(memory) < BATCH_SIZE:
        return

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    for _ in range(no_of_epochs):
        # print("in training loop")
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action) 
        reward_batch = torch.cat(batch.reward)
        # print(batch.state[0], batch.state[0].size())
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q valuesop
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    policy_net.train(False)

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 100

rewards = []
c = 1
eps = 1.0
min_eps = 0.00
for i_episode in range(1, num_episodes + 1):
    print(f"Episode {i_episode} simulating...")

    # Saving of checkpoints
    if i_episode % 5 == 0:
        exist1 = os.path.exists(f"./DDQN/runs/{network}/run_{run}/policy_net_checkpoint_{c}")
        if not exist1:
            os.makedirs(f"./DDQN/runs/{network}/run_{run}/policy_net_checkpoint_{c}")

        exist2 = os.path.exists(f"./DDQN/runs/{network}/run_{run}/target_net_checkpoint_{c}")
        if not exist2:
            os.makedirs(f"./DDQN/runs/{network}/run_{run}/target_net_checkpoint_{c}")

        for id in env.agentIDs:
            policy_net = models[id]["policy_net"]
            target_net = models[id]["target_net"]
            torch.save(policy_net.state_dict(),
                       f"./DDQN/runs/{network}/run_{run}/policy_net_checkpoint_{c}/agent_{id}")
            torch.save(target_net.state_dict(),
                       f"./DDQN/runs/{network}/run_{run}/target_net_checkpoint_{c}/agent_{id}")
        c += 1

    state = env.reset(traffic_generator.generate_routefile, i_episode)

    guesses = 0
    # Initialize the environment and get it's state
    states = {}
    for id, obs in zip(env.agentIDs, state):
        states[id] = torch.tensor(
            obs, dtype=torch.float32, device=device).unsqueeze(0)

    # if i_episode % 5 == 0 :
    #     torch.save(policy_net.state_dict(), f"./models/DQNv2/policy_net_checkpoint_{c}")
    #     torch.save(target_net.state_dict(), f"./models/DQNv2/target_net_checkpoint_{c}")
    #     c += 1

    eps_reward = 0
    t = 1

    
    for t in count():
        #print(f"Episode {i_episode} Step {t}")
        # print(t)
        actions = {}
        step_actions = []

        for id in env.agentIDs:
            action = select_action(models[id], states[id], eps)
            actions[id] = action
            step_actions.append(action.item())

        observations, reward, terminated, truncated = env.step(step_actions)

        # ind = env.agentIDs.index("232")
        # print(env.info["232"])
        # print(observations[ind], reward[ind], step_actions[ind])

        # for id in env.agentIDs:
        #     optimize_model(models[id], 5)  

        done = True in terminated
        next_states = {}
        rewards = {}

        for i, id in zip(range(len(observations)), env.agentIDs):
            next_states[id] = torch.tensor(
                observations[i], dtype=torch.float32, device=device).unsqueeze(0)
            rewards[id] = torch.tensor([reward[i]], device=device)
            models[id]['memory'].push(
                states[id], actions[id], next_states[id], rewards[id])
            eps_reward += rewards[id].item()

        # Move to the next state
        states = next_states

        # Perform one step of the optimization (on the policy network)

        if done:
            episode_durations.append(t + 1)
            break

    print("Training...")
    for id in env.agentIDs:
        optimize_model(models[id], 1800)
        target_net_state_dict = models[id]['target_net'].state_dict()
        policy_net_state_dict = models[id]['policy_net'].state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        models[id]['target_net'].load_state_dict(target_net_state_dict)
    print("Training done!")
    
    # if i_episode % 5 == 0:
    #     target_net_state_dict = models[id]['target_net'].state_dict()
    #     policy_net_state_dict = models[id]['policy_net'].state_dict()
    #     for key in policy_net_state_dict:
    #         target_net_state_dict[key] = policy_net_state_dict[key] * \
    #             TAU + target_net_state_dict[key]*(1-TAU)
    #     models[id]['target_net'].load_state_dict(target_net_state_dict)

    avg_eps_reward = eps_reward/t
    # writer.add_scalar("charts/avg_episodic_return", avg_eps_reward, i_episode)

    # logging.info(
    #     f"Episode {i_episode} | Reward : {avg_eps_reward} | Guesses : {guesses} | Total timesteps : {t}")
    print(f"Episode {i_episode} | Reward : {avg_eps_reward} | Acc. waiting time : {round(env.getAvgAccumulatedWaitingTime(), 3)} | Guesses : {guesses} | Epsilon : {eps} | Total timesteps : {t}")

    with open(f"./DDQN/runs/{network}/run_{run}/logs.csv", "a") as f:
        print(f"{i_episode},{avg_eps_reward},{round(env.getAvgAccumulatedWaitingTime(), 3)},{round(env.getMaxAccumulatedWaitingTime(), 3)},{eps}", file=f)

    if eps > min_eps:
        eps = eps - (1/num_episodes)
        # eps = eps*0.95

exist1 = os.path.exists(f"./DDQN/runs/{network}/run_{run}/policy_net")
if not exist1:
    os.makedirs(f"./DDQN/runs/{network}/run_{run}/policy_net")

exist2 = os.path.exists(f"./runs/{network}/run_{run}/target_net")
if not exist2:
    os.makedirs(f"./runs/{network}/run_{run}/target_net")

for id in env.agentIDs:
    policy_net = models[id]["policy_net"]
    target_net = models[id]["target_net"]
    torch.save(policy_net.state_dict(), f"./DDQN/runs/{network}/run_{run}/policy_net/agent_{id}")
    torch.save(target_net.state_dict(), f"./DDQN/runs/{network}/run_{run}/target_net/agent_{id}")
