from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import numpy as np
import traci
# import libsumo as traci
import os 
import os, sys
import pandas as pd
import time
import random
from traci._trafficlight import Logic
from typing import List

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


import sumolib


class SumoEnvironment(gym.Env):

    def __init__(self,dim = 5,max_steps = 5000,cfg_file = "./most.sumocfg", use_gui = False, results = False, rPath = None, run_name = None):
        # if 'SUMO_HOME' in os.environ:
        #     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #     sys.path.append(tools)
        # else:   
        #     sys.exit("please declare environment variable 'SUMO_HOME'")
        self.run_name = run_name
        self.rPath = rPath
        self.results = results
        self.episode = 0
        self._step = 0
        self.max_steps = max_steps
        self._cfg = cfg_file
        self.start = False
        # self.use_gui = use_gui
        self.use_gui = True
        self.sumo_seed = "random"
        self.label = 0
        self.delta_time = 10
        self.yellow_time = 2
        # self.init_agents_info()

    

    def init_agents_info(self):

        self.agentIDs = list(self.sumo.trafficlight.getIDList())
        self.agents_n = len(self.agentIDs)
        self.info = {}
        self.action_spaces = []
        for id in self.agentIDs :
            currentProgram = self.sumo.trafficlight.getProgram(id)
            programs : List[Logic]   = self.sumo.trafficlight.getAllProgramLogics(id)
            program_names = [p.programID for p in programs]

            # print(currentProgram, programs, program_names)

            index = program_names.index(currentProgram)

            phases = programs[index].getPhases()
            phase_encodings = []
            for p in phases :
                if 'y' not in p.state and ('g'  in p.state or 'G' in p.state) and  p.state not in phase_encodings:
                    phase_encodings.append(p.state)


            lanes = self.sumo.trafficlight.getControlledLanes(id)
            total_phases = len(phase_encodings)
            
            self.action_spaces.append(Discrete(total_phases))
            self.info[id] = {"Phases" : phase_encodings, "Incoming Lanes" :lanes, "State Space" : total_phases *2, "Action Space" : total_phases  }

        self.trafficLights = {ts: TrafficSignal(ts, self.sumo.trafficlight, self.sumo, self.info[ts]["Phases"] ) for ts in self.agentIDs}

    def reset(self, callback, seed):
        self._step = 0
        self.episode += 1
        if self.start:
            self.sumo.close()
        else :
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

        if self.results :
            sumo_cmd.extend(["--summary",
            f"./results/{self.rPath}/Summary.xml",
            "--queue-output",
            f"./results/{self.rPath}/QueueInfo.xml",
            "--tripinfo-output",
            f"./results/{self.rPath}/VehicleInfo.xml"])

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.use_gui :
            sumo_cmd.extend(["--start", "--quit-on-end"])

        self.label += random.randint(0,5000)
        traci.start(sumo_cmd, label=self.label)
        self.sumo : traci = traci

        if self.use_gui :
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        

    def sumo_steps(self,n):
        for x in range(n) :
            self._step += 1
            self.sumo.simulationStep()
    
    def step(self,action):
        start_time = time.time()

        # initial_state = self.get_state()
        self.apply_actions(action,"yellow")
        self.sumo_steps(self.yellow_time)
        self.apply_actions(action,"green")
        self.sumo_steps(self.delta_time - self.yellow_time)
        dones = [False for tid in self.agentIDs]
        if self._step > self.max_steps and self.sumo.vehicle.getIDCount() == 0:
            dones = [True for tid in self.agentIDs]


        # print(f"Step Time : {time.time() - start_time}")
        rewards = self.measure_reward()
        observations = self.get_state()
        # rewards = [2,32,32,22332]
        # observations = [2,323,21,32]
        # print(f"State Time : {time.time() - start_time}")
        global_reward = np.sum(rewards)
        # infos = {tid : {"t_info" : obs } for tid,obs in observations }
        truncateds = {tid : False for tid in self.agentIDs}

        # if self.run_name != None :
        #     self.logging(action,initial_state,observations,rewards)

        
        return observations, rewards, dones,truncateds


    def get_state(self):
        state = []
        for tid in self.agentIDs:
            state.append(self.trafficLights[tid].getState())
        return state

    def measure_reward(self):
        rewards = {}
        for ts in self.agentIDs:
            rewards[ts] = {
                "halted_vehicles" : self.trafficLights[ts].getHaltedCars(),
                "wait_times" : self.trafficLights[ts].totalMaxWaitingTime()
            }
        
        # halted_vehicles = {ts.id : ts.getHaltedCars() for ts in self.trafficLights.values()}
        # wait_times = {ts.id : ts.totalMaxWaitingTime() for ts in self.trafficLights.values()}
        # total_queue = sum(halted_vehicles.values())
        # total_wait = sum(wait_times.values())


        t_rewards = [(r["halted_vehicles"] + 0.2*r["wait_times"]) for tid,r in rewards.items()]
        
        return t_rewards

    def apply_actions(self, actions : list,phase):
        for i in range(len(self.agentIDs)):
                self.trafficLights[self.agentIDs[i]].setPhase(actions[i],phase)

    def logging(self,actions, prev_state,new_state,rewards):
        tIDs = self.agentIDs
        GreenPhases = [self.trafficLights[tIDs[0]].getPhaseInfo(actions[x]) for x in range(len(actions))]

        junctionInfo = {id : {
            "episode" : self.episode,
            "timestep" : self._step - self.delta_time,
            "Action" : action,
            "Reward" : reward,
            "PreviousState" : prev_state,
            "NewState" : new_state,

        } for id,action,prev_state,new_state,reward in zip(tIDs,GreenPhases, prev_state,new_state,rewards)}

        data = []
        for id, info in junctionInfo.items():
            row_data = {"Episode" : info["episode"] ,"Timestep" : info["timestep"] ,"tid": id, "Action": info["Action"],"Reward" : info["Reward"], "Previous State" : info["PreviousState"],"New State" : info["NewState"]}
            data.append(row_data)
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file with append mode
        with open(f"./logging/{self.run_name}.csv", 'a') as f:
            df.to_csv(f, index=False, header=f.tell() == 0, sep=';')

        return junctionInfo
        



class TrafficSignal:

    def __init__(self, id,tl_object : traci.trafficlight, sumo : traci, Phases : list ) :

        self.phases = Phases
        self.id = id
        self.tl = tl_object
        self.lanes, self.incoming_lanes = self.getIncomingLanes()

        self.Validate()

        self.sumo = sumo


    def Validate(self):
        for p in self.phases:
            # print(len(p), len(self.lanes),len(self.incoming_lanes))
            assert len(p) == len(self.lanes)
        

    def getIncomingLanes(self):
        lanes = self.tl.getControlledLanes(self.id)
        incoming_lanes = []
        [incoming_lanes.append(x) for x in lanes if x not in incoming_lanes]
        return lanes,incoming_lanes
    
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






# env = SumoEnvironment(use_gui = False, max_steps = 3600)
# env.reset()
# i = 0
# #i increases every delta seconds
# while i < 1000:
#     i +=1
#     action = i%5
#     actions = {agent : action for agent in env.agents.keys()}
#     print()
#     print(f"Step {i}")
#     print(f"Action {env.trafficLights['nt1'].getPhaseInfo(action)}")
#     print()
#     env.step(actions)




# simple_trainer = DQNConfig().environment(env=SumoEnvironment).rollouts(num_rollout_workers=1).build()
# simple_trainer.train()



# check_env(env)
