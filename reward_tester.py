import numpy as np
from TD3_agent import *

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([-2, -2 , 0])
        }

approach_direction = "pos_R-bar"    # choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
reward_type = "simple"           # choose from simple, full or ...
rwd_comp = RewardComputer(approach_direction, reward_type, docking_port_locations)
state1 = np.array([0,-10,0,0,1,0])
state2 = np.array([0,-8,0,0,1,0])
action1 = np.array([1,1,1])
action2 = np.array([0,0,0])

rwd_comp.get_reward(state1, action1)
rwd_comp.get_reward(state2, action2)