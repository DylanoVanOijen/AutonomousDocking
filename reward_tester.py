import numpy as np
from TD3_agent import *

approach_direction = "pos_R-bar"    # choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
reward_type = "simple"           # choose from simple, full or ...
rwd_comp = RewardComputer(approach_direction, reward_type)
state = np.array([0,-10,0,0,1,0])
state = np.array([0,-8,0,0,1,0])
rwd_comp.get_reward(state)