import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import traci
# import libsumo as traci
import sumolib

import os, sys
import pandas as pd
import time
import random
from typing import List

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

class SumoEnvironment(gym.Env):
    def __init__(self, max_steps, cfg_file, use_gui, results, delta_time, yellow_time):
        self.max_steps = max_steps
        self.cfg_file = cfg_file
        self.use_gui = use_gui
        self.results = results
        self.episode = 0
        self.timestep = 0

    def init_agents(self):
        self.agents = list(self.sumo.trafficlight.getIDList())
        self.agents_n = len(self.agentIDs)