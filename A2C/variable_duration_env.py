import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
# import traci
import libsumo as traci
import sumolib
from sumo_env_new_state_space import TrafficSignal

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
    def __init__(self, max_steps, cfg_file, use_gui):
        self.max_steps = max_steps
        self._cfg = cfg_file
        self.use_gui = use_gui
        self.results = True
        self.episode = 0
        self.timestep = 0
        self.sumo = traci
        self.start = False
        self.sumo_seed = "random"
        self.label = 0
        self.delta_time = 20
        self.yellow_time = 2

    def reset(self, callback, seed):
        self._step = 0
        self.episode += 1

        if self.start:
            self.sumo.close()
        else:
            self.start = True

        callback(seed)

        self._start_simulation()
        self.init_agents_info()

        obs = self.get_state()
        return obs

    def _start_simulation(self):
        if self.use_gui :
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        
        sumo_cmd = [
            self._sumo_binary,
            # "-n",
            # self._net,
            # "-r",
            # self._route,
            # "--max-depart-delay",
            # str(self.max_depart_delay),
            # "--waiting-time-memory",
            # str(self.waiting_time_memory),
            "-c",
            self._cfg,
            "--no-warnings"
        ]

        # if self.results :
        #     sumo_cmd.extend(["--summary",
        #     f"./results/{self.rPath}/Summary.xml",
        #     "--queue-output",
        #     f"./results/{self.rPath}/QueueInfo.xml",
        #     "--tripinfo-output",
        #     f"./results/{self.rPath}/VehicleInfo.xml"])

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.use_gui :
            sumo_cmd.extend(["--start", "--quit-on-end"])

        self.label += random.randint(0,5000)
        traci.start(sumo_cmd, label=self.label)

        if self.use_gui :
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def sumo_steps(self, n):
        for _ in range(n):
            self._step += 1
            self.sumo.simulationStep()

    def get_state(self):
        state = []
        for tid in self.agentIDs:
            state.append(self.trafficLights)

    def init_agents_info(self):
        self.agentIDs = list(self.sumo.trafficlight.getIDList())
        self.agents_n = len(self.agentIDs)
        self.info = {}
        self.action_spaces = []

        for ids in self.agentIDs:
            currentProgram = self.sumo.trafficlight.getProgram(ids)
            programs = self.sumo.trafficlight.getAllProgramLogics(ids)
            program_names = [p.programID for p in programs]

            index = program_names.index(currentProgram)

            phases = programs[index].getPhases()
            phase_encodings = []

            for p in phases:
                if 'y' not in p.state and ('g'  in p.state or 'G' in p.state) and  p.state not in phase_encodings:
                    phase_encodings.append(p.state)

            lanes = self.sumo.trafficlight.getControlledLanes(ids)
            total_phases = len(phase_encodings)

            self.action_spaces.append(Discrete(total_phases))
            self.info[ids] = {"Phases" : phase_encodings, "Incoming Lanes" :lanes, "State Space" : total_phases *2, "Action Space" : total_phases  }

        self.trafficLights = {ts: TrafficSignal(ts, self.sumo.trafficlight, self.sumo, self.info[ts]["Phases"] ) for ts in self.agentIDs}

    def get_state(self):
        state = []
        for ids in self.agentIDs:
            state.append(self.trafficLights[ids].getState())
        return state
    
    def get_reward(self):
        # reward = - sum of cars in all lanes around signal
        rewards = []
        for ts in self.agentIDs:
            rewards.append(-self.trafficLights[ts].getTotalCarsInLanes())

        return rewards
            
    def apply_action(self, actions, phase):
        for i in range(len(self.agentIDs)):
            self.trafficLights[self.agentIDs[i]].setPhase(actions[i], phase)

    def step(self, action):
        # Change this
        # Take one step forward
        # First apply actions for required traffic lights, wherever there is a change
        self.sumo_steps(1)
        for i in action:
            if i is not None:
                self.apply_action(id, action, phase)
                # apply action must combine a set time for yellow, and only then put green
                # apply action must also work only on a single junction (i.e. for each id)
        # Step until there is at least one traffic light that requires state change
        while self._step not in self.next_state_change_step:
            self.sumo.simulationStep()
        # now construct list of state, action, etc. for only the changed models
        # return them
        # return observations, rewards, dones, truncateds