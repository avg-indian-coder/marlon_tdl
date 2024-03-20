from DDQN.generator import TrafficGen

gen = TrafficGen("nets/bo/joined_buslanes.net.xml", "nets/bo/generated_route.rou.xml", 3600, 1000, 0.1)
print(gen.generate_routefile(1))