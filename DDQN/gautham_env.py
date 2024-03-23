import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
# import traci
import libsumo as traci
import sumolib
# from sumo_env_new_state_space import TrafficSignal

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

reward_functions = [
    "carsInLanes",
    ""
]

class SumoEnvironment(gym.Env):
    def __init__(self, max_steps, neighbours, degree_of_multiagency, network, cfg_file, eval_cfg, use_gui, reward_function):
        self.max_steps = max_steps
        self._cfg = cfg_file
        self._eval_cfg = eval_cfg
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
        self.run = 1
        self.network = network
        self.all_vehicles_acc_waiting_time = dict()
        self.df = None
        self.neighbours = neighbours
        self.neighbouring_nodes = dict()
        self.net = sumolib.net.readNet(f"./nets/{self.network}/network.net.xml")
        self.degree = degree_of_multiagency
        # self.net.getNode()

    def updateAccumulatedWaitingTime(self):
        vehicles = self.sumo.vehicle.getIDList()

        for vehicle in vehicles:
            self.all_vehicles_acc_waiting_time[vehicle] = self.sumo.vehicle.getAccumulatedWaitingTime(vehicle)

    def getAvgAccumulatedWaitingTime(self):
        return sum(self.all_vehicles_acc_waiting_time.values())/len(self.all_vehicles_acc_waiting_time)
    
    def getMaxAccumulatedWaitingTime(self):
        return max(self.all_vehicles_acc_waiting_time.values())

    def get_run(self):
        while True:
            if os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}"):
                self.run += 1
            else:
                break
        
        if not os.path.exists(f"./DDQN/runs/{self.network}"):
            os.mkdir(f"./DDQN/runs/{self.network}")
        os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}")
        # os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/logs")
        # with open(f"./DDQN/runs/{self.network}/run_{self.run}/logs.csv", "a") as f:
        #     print("Episode,")

        return self.run

    def reset(self, callback, seed, evaluate: bool):
        # sumolib.net.TLS.getConnections()
        if evaluate:
            self._step = 0

            if self.start:
                # self.get_adjacent_nodes(1)
                self.sumo.close()
            else:
                self.start = True
            
            self._start_evaluation()
            self.init_agents_info()
            
            obs = self.get_state()
            return obs
        
        self._step = 0
        self.episode += 1
        # self.df = pd.DataFrame(columns=["step","waiting_time"])

        if self.start:
            # self.get_adjacent_nodes(1)
            self.sumo.close()
        else:
            self.start = True

        callback(seed)

        self._start_simulation()
        self.init_agents_info()

        obs = self.get_state()
        return obs
    
    def _start_evaluation(self):
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
            self._eval_cfg,
            "--no-warnings"
        ]

        # sumo_cmd.extend(["--summary",
        # f"./DDQN/esults/{self.rPath}/Summary.xml",
        # "--queue-output",
        # f"./results/{self.rPath}/QueueInfo.xml",
        # "--tripinfo-output",
        # f"./results/{self.rPath}/VehicleInfo.xml"])
        if not os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}/Summary_Eval"):
            os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/Summary_Eval")
        if not os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}/QueueInfo_Eval"):
            os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/QueueInfo_Eval")
        if not os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}/VehicleInfo_Eval"):
            os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/VehicleInfo_Eval")

        sumo_cmd.extend(["--summary",
                         f"./DDQN/runs/{self.network}/run_{self.run}/Summary_Eval/episode_{self.episode}.xml",
                         "--queue-output",
                         f"./DDQN/runs/{self.network}/run_{self.run}/QueueInfo_Eval/episode_{self.episode}.xml",
                         "--tripinfo-output",
                         f"./DDQN/runs/{self.network}/run_{self.run}/VehicleInfo_Eval/episode_{self.episode}.xml"])

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.use_gui :
            sumo_cmd.extend(["--start", "--quit-on-end"])

        # to get accumulated waiting time from simulation start
        sumo_cmd.extend(["--waiting-time-memory", "10000"])

        self.label += random.randint(0,5000)
        traci.start(sumo_cmd, label=self.label)

        if self.use_gui :
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

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

        # sumo_cmd.extend(["--summary",
        # f"./DDQN/esults/{self.rPath}/Summary.xml",
        # "--queue-output",
        # f"./results/{self.rPath}/QueueInfo.xml",
        # "--tripinfo-output",
        # f"./results/{self.rPath}/VehicleInfo.xml"])
        if not os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}/Summary"):
            os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/Summary")
        if not os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}/QueueInfo"):
            os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/QueueInfo")
        if not os.path.exists(f"./DDQN/runs/{self.network}/run_{self.run}/VehicleInfo"):
            os.mkdir(f"./DDQN/runs/{self.network}/run_{self.run}/VehicleInfo")

        sumo_cmd.extend(["--summary",
                         f"./DDQN/runs/{self.network}/run_{self.run}/Summary/episode_{self.episode}.xml",
                         "--queue-output",
                         f"./DDQN/runs/{self.network}/run_{self.run}/QueueInfo/episode_{self.episode}.xml",
                         "--tripinfo-output",
                         f"./DDQN/runs/{self.network}/run_{self.run}/VehicleInfo/episode_{self.episode}.xml"])

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.use_gui :
            sumo_cmd.extend(["--start", "--quit-on-end"])

        # to get accumulated waiting time from simulation start
        sumo_cmd.extend(["--waiting-time-memory", "10000"])

        self.label += random.randint(0,5000)
        traci.start(sumo_cmd, label=self.label)

        if self.use_gui :
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def sumo_steps(self, n):
        for _ in range(n):
            self._step += 1
            self.sumo.simulationStep()
            self.updateAccumulatedWaitingTime()

    """ def get_state(self):
        state = []
        for tid in self.agentIDs:
            state.append(self.trafficLights) """

    def init_agents_info(self):
        self.agentIDs = list(self.sumo.trafficlight.getIDList())
        # for ids in self.agentIDs:
        # print("a0", sorted([n.getID() for n in self.net.getNode("a0").getNeighboringNodes()]))
        # sample_tls = sumolib.net.TLS(self.agentIDs[0])
        # print("connections")
        # print(sample_tls.getConnections())
        self.agents_n = len(self.agentIDs)
        self.info = {}
        self.action_spaces = []
        # print(self.sumo.trafficlight.getIDList())
        for ids in self.agentIDs:
            currentProgram = self.sumo.trafficlight.getProgram(ids)
            programs = self.sumo.trafficlight.getAllProgramLogics(ids) # all the traffic lights in env  
            program_names = [p.programID for p in programs]

            # print(programs)


            index = program_names.index(currentProgram)

            phases = programs[index].getPhases()
            # print(list(set(self.sumo.trafficlight.getControlledLanes(ids))))
            # print(programs)
            # print("")
            # print(self.sumo.trafficlight.getIDList())
            # print(len(phases))
            phase_encodings = []

            for p in phases:
                if 'y' not in p.state and ('g'  in p.state or 'G' in p.state) and  p.state not in phase_encodings:
                    phase_encodings.append(p.state)

            # print(ids ,phase_encodings)
            lanes = self.sumo.trafficlight.getControlledLanes(ids)
            total_phases = len(phase_encodings)

            self.action_spaces.append(Discrete(total_phases))
            # self.info[ids] = {"Phases" : phase_encodings, "Incoming Lanes" :lanes, "State Space" : total_phases *2, "Action Space" : total_phases  }
            # self.info[ids] = {"Phases" : phase_encodings, "Incoming Lanes" :lanes, "State Space" : len(set(lanes)), "Action Space" : total_phases  }
            self.info[ids] = {"Phases" : phase_encodings, "Incoming Lanes" :lanes, "Action Space" : total_phases  }

        self.trafficLights = {ts: TrafficSignal(ts, self.sumo.trafficlight, self.sumo, self.info[ts]["Phases"], self.neighbours[ts], self.degree ) for ts in self.agentIDs}
        for ids in self.agentIDs:
            # self.info[ids]["State Space"] =  len(self.get_state())
            self.info[ids]["State Space"] = len(self.trafficLights[ids].getState())
        

    def get_adjacent_nodes(self):
        self.edge_to_node = dict()
        lanes = dict()
        # for ids in self.agentIDs:
            # lanes[ids] = sorted(list(set(self.sumo.trafficlight.getControlledLinks(ids))))
            # lanes[ids] = self.sumo.trafficlight.getControlledLinks(ids)

        # for light in self.net.getTrafficLights():
            # print(light.getID(),light.getConnections())
        # print(len(self.net.getTrafficLights()))
        # print(len(self.net.getNodes()))
        # print(lanes)

        for ids, conn_lanes in lanes.items():
            for lane in conn_lanes:
                if lane not in self.edge_to_node:
                    self.edge_to_node[lane] = [ids, ]
                else:
                    self.edge_to_node[lane].append(ids)
        # print(self.edge_to_node)
        self.neighbour_nodes = dict()

        for _, ids_list in self.edge_to_node.items():
            for i in range(len(ids_list)):
                for j in range(i+1, len(ids_list)):
                    if ids_list[i] not in self.neighbour_nodes:
                        self.neighbour_nodes[ids_list[i]] = [ids_list[j]]
                    else:
                        self.neighbour_nodes[ids_list[i]].append(ids_list[j])


        # print(self.neighbour_nodes)

        


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
        self.apply_action(action, "yellow")
        self.sumo_steps(self.yellow_time)
        self.apply_action(action, "green")
        self.sumo_steps(self.delta_time - self.yellow_time)

        dones = [False for _ in self.agentIDs]
        if self._step > self.max_steps and self.sumo.vehicle.getIDCount() == 0:
            dones = [True for _ in self.agentIDs]

        rewards = self.get_reward()
        observations = self.get_state()
        truncateds = {tid : False for tid in self.agentIDs}

        return observations, rewards, dones, truncateds
    
    def getAvgWaitingTime(self):
        # Waiting time of all vehicles at an instant
        vehicles = self.sumo.vehicle.getIDList()
        avg_waiting_time = 0

        for vehicle in vehicles:
            avg_waiting_time += self.sumo.vehicle.getAccumulatedWaitingTime(vehicle)
        
        return avg_waiting_time/len(vehicles)
    
    def getMaxWaitingTime(self):
        # Maximum waiting time of all vehicles at an instant
        vehicles = self.sumo.vehicle.getIDList()
        max_waiting_time = 0

        for vehicle in vehicles:
            # avg_waiting_time += self.sumo.vehicle.getAccumulatedWaitingTime(vehicle)
            acc_wait_time = self.sumo.vehicle.getAccumulatedWaitingTime(vehicle)
            if acc_wait_time > max_waiting_time:
                max_waiting_time = acc_wait_time
        
        return max_waiting_time
    
    def getMinWaitingTime(self):
        # Maximum waiting time of all vehicles at an instant
        vehicles = self.sumo.vehicle.getIDList()
        min_waiting_time = 2**16

        for vehicle in vehicles:
            # avg_waiting_time += self.sumo.vehicle.getAccumulatedWaitingTime(vehicle)
            acc_wait_time = self.sumo.vehicle.getAccumulatedWaitingTime(vehicle)
            if acc_wait_time < min_waiting_time:
                min_waiting_time = acc_wait_time
        
        return min_waiting_time

    # def logging(self):
    #     # save characteristics during an episode
    #     data = []

    #     with open(f"./DDQN/runs/{self.network}/run_{self.run}/logs.csv", "a") as f:
    #         if self.
    #         print(f"")


        


        

    

class TrafficSignal:

    def __init__(self, id,tl_object : traci.trafficlight, sumo : traci, Phases : list, neighbours : list, degree: int ) :

        self.phases = Phases
        self.id = id
        self.tl = tl_object
        self.neighbours = neighbours
        self.degree = degree
        self.lanes, self.incoming_lanes = self.getIncomingLanes()
        # self.Validate()

        self.sumo = sumo


    def Validate(self):
        for p in self.phases:
            # print(len(p), len(self.lanes),len(self.incoming_lanes))
            assert len(p) == len(self.lanes)
        

    def getIncomingLanes(self):
        lanes = list(self.tl.getControlledLanes(self.id))
        if self.degree == 1:
            for ids in self.neighbours:
                lanes.extend(list(self.tl.getControlledLanes(ids)))
        incoming_lanes = []
        [incoming_lanes.append(x) for x in lanes if x not in incoming_lanes]
        # print(incoming_lanes)
        return lanes, sorted(incoming_lanes)
    
    def getHaltedCars(self):
        num_cars = 0
        for lane in self.incoming_lanes:
            num_cars += self.sumo.lane.getLastStepHaltingNumber(lane)
        
        return num_cars
    
    def getState(self):
        lanes = self.incoming_lanes
        state = []

        for i in range(len(lanes)):
            vehicle_count = self.sumo.lane.getLastStepVehicleNumber(lanes[i])
            state.append(vehicle_count)

        return state


    def getNewState(self):

        phases : List[str] = self.phases
        lanes = self.incoming_lanes
        state = []
        laneVehicles = []
        waitingTimes = []

        # In each lane calculation
        vehicles_array = []
        times_array = []

        for i in range(len(lanes)) :
            if i!= 0 and lanes[i] == lanes[i - 1]:
                vehicles_array.append(vehicles_array[i - 1])
                times_array.append(times_array[i - 1])

            else :
                lane = lanes[i]
                vehicles = self.sumo.lane.getLastStepVehicleNumber(lane)
                vehicles_array.append(vehicles)

                car_wait = 0
                max_pos = 0
                cur_cars = self.sumo.lane.getLastStepVehicleIDs(lane)
                for vid in cur_cars:
                    car_pos = self.sumo.vehicle.getLanePosition(vid)
                    if car_pos > max_pos:
                        max_pos = car_pos
                        car_wait = self.sumo.vehicle.getWaitingTime(vid)

                times_array.append(car_wait) 

        for phase in phases :
            added = []
            vehicles = 0
            waits = 0
            for i in range(len(phase)):
                if phase[i].upper() == "G" and lanes[i] not in added :
                    lane = lanes[i]
                    vehicles += vehicles_array[i]
                    waits += times_array[i]


                    added.append(lane)
            
            laneVehicles.append(vehicles)
            waitingTimes.append(waits)

        for x,y in zip(laneVehicles,waitingTimes):
            state.append(x)
            state.append(y)

        return state
    
    def setPhase(self, next_phase_index,phase):
        next_phase = self.phases[next_phase_index]
        self.current_index = next_phase_index
        self.current_phase = self.phases[next_phase_index]
        if phase == "yellow" :
            current_phase = self.tl.getRedYellowGreenState(self.id)
            if current_phase != next_phase :
                immediate_phase = ""
                for c,n in zip(current_phase,next_phase) :
                    if c!= n :
                        immediate_phase += "y"
                    else :
                        immediate_phase += c
            else :
                return
            
        if phase == "green":
            immediate_phase = next_phase

        self.tl.setRedYellowGreenState(self.id,immediate_phase)

    def getTotalCarsInLanes(self):
        # Returns all the cars in each lane of each intersection
        # _, incoming_lanes = self.getIncomingLanes()
        total_cars_around_intersection = 0

        for lane in self.incoming_lanes:
            no_cars_in_lane = len(self.sumo.lane.getLastStepVehicleIDs(lane))
            total_cars_around_intersection += no_cars_in_lane

        return total_cars_around_intersection

    def totalMaxWaitingTime(self):
        # Returns sum of max wait time of each lane
        total_mwt = 0
        for lane in self.incoming_lanes:
            car_wait = 0
            max_pos = 0
            cur_cars = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in cur_cars:
                car_pos = self.sumo.vehicle.getLanePosition(vid)
                if car_pos > max_pos:
                    max_pos = car_pos
                    car_wait = self.sumo.vehicle.getWaitingTime(vid) 
            total_mwt += car_wait
        return total_mwt

