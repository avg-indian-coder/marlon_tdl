import numpy as np
import math
import os
import platform
import random
import subprocess

def generator(net, route, end_time, no_vehicles, p):
    '''
    Generate the route file based on above the parameters
    '''
    if platform.system() == 'Windows':
        cmd_code = f'python "%SUMO_HOME%/tools/randomTrips.py" --validate -r {route} --end {no_vehicles} -n {net}'
    else:
        cmd_code = f'SUMO_HOME/tools/randomTrips.py --validate -r {route} --end {no_vehicles} -n {net}'
    # os.system(cmd_code)
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call([cmd_code], stdout=devnull, stderr=subprocess.STDOUT)

    f = open(route, "r+")
    l = f.readlines()

    for i in range(len(l)):
        if "vehicle" in l[i]:
            line_idx = i
            break

    vehicle_count = len(l[line_idx + 1:]) / 3 # count of the vehicles in the route

    # get a weibull distribution too now, assume the simulation starts at 0 and ends at end_time
    timings = np.random.weibull(2, int(vehicle_count))
    timings = np.sort(timings)

    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    min_new = 0
    max_new = end_time
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

    car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

    # car_gen_steps now contains the sorted times according to the Weibull distribution
    # now the job is to replace the time-steps in the xml file with the ones in car_gen_steps

    for step in car_gen_steps:
        if random.random() < p:
            l[line_idx] = l[line_idx][:l[line_idx].find('depart')] + f'depart="{step}" type="type2"' + '>\n'
        else:
            l[line_idx] = l[line_idx][:l[line_idx].find('depart')] + f'depart="{step}" type="type1"' + '>\n'
        line_idx += 3

    f.close()

    f = open(route, "w")
    f.writelines(l)

    return car_gen_steps

class TrafficGen:
    def __init__(self, net, route, end_time, no_vehicles, p):
        self._net = net
        self._route = route
        self._end_time = end_time
        self._no_vehicles = no_vehicles
        self._p = p
    
    def generate_routefile(self, seed, weibull=True):
        np.random.seed(seed)
        # print(seed)

        if platform.system() == 'Windows':
            cmd_code = f'python3 "%SUMO_HOME%/tools/randomTrips.py" --seed {seed} --validate -r {self._route} --end {self._no_vehicles} -n {self._net}'
        else:
            cmd_code = f'$SUMO_HOME/tools/randomTrips.py --seed {seed} --validate -r {self._route} --end {self._no_vehicles} -n {self._net} > /dev/null'

        # print(cmd_code)
        os.system(cmd_code)
        # with open(os.devnull, 'wb') as devnull:
        #     subprocess.run([cmd_code], shell=True, stdout=devnull, stderr=subprocess.STDOUT)
        # print("Gay")
        # subprocess.check_output(cmd_code, shell=True, stderr=subprocess.STDOUT)

        # if not weibull:
        #     return

        f = open(self._route, "r+")
        l = f.readlines()

        for i in range(len(l)):
            if "vehicle" in l[i]:
                line_idx = i
                insert_idx = i
                break
        
        vtype = '<vType id="type1" length="5" accel="5" decel="10" />\n<vType id="type2" color="1,0,0" length="7" accel="5" decel="10" vClass="emergency" />\n'

        vehicle_count = len(l[line_idx + 1:]) / 3 # count of the vehicles in the self._route

        # get a weibull distribution too now, assume the simulation starts at 0 and ends at self._end_time
        if weibull:
            timings = np.random.weibull(2, int(vehicle_count))
        else:
            timings = np.random.randn(int(vehicle_count))
        timings = np.sort(timings)

        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._end_time
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # car_gen_steps now contains the sorted times according to the Weibull distribution
        # now the job is to replace the time-steps in the xml file with the ones in car_gen_steps

        for step in car_gen_steps:
            if random.random() < self._p:
                l[line_idx] = l[line_idx][:l[line_idx].find('depart')] + f'depart="{step}" type="type2"' + '>\n'
            else:
                l[line_idx] = l[line_idx][:l[line_idx].find('depart')] + f'depart="{step}" type="type1"' + '>\n'
            line_idx += 3

        l.insert(insert_idx, vtype)

        f.close()

        f = open(self._route, "w")
        f.writelines(l)

        # return car_gen_steps


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    generator = TrafficGen('nets/bo/network.net.xml', 'nets/3x3/evaluation_route.rou.xml', 5400, 2000, 0.1)
    generator.generate_routefile(3, weibull=False)

    # if True:
    #     plt.title("Weibull distribution (2000 cars generated)")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("Number of cars generated")
    #     plt.hist(car_gen_steps, bins = 54)
    #     plt.show()