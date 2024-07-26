import numpy as np
from TD3_agent import *

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([-2, -2 , 0])
        }

docking_settings = {
    "KOS_size" : 25,
    "corridor_base_radius": 1,   # meter
    "corridor_angle" : np.deg2rad(20), 
    "max_distance" : 50,
    "max_offdir_pos" : 0.2,
    "max_offdir_vel" : 0.01,
    "ideal_dir_vel" : 0.2,
    "min_dir_vel" : 0.1,
    "max_dir_vel" : 0.2  
}

reward_parameters = {
    "eta" : 10,
    "kappa" : 0.1,  # tanh scaling
    "lamda" : 1.0,  # no action reward scaling
    "mu" : 0.05,
    "corridor_penalty" : 0,
    "far_away_penalty" : 0,
    "docking_pos_bonus" : 1000,
    "docking_vel_bonus" : 1000,
    "docking_pos_bonus_scaling" : 1000,
    "docking_vel_bonus_scaling" : 1000,
}

approach_direction = "pos_R-bar"    # choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
reward_type = "simple"           # choose from simple, full or ...
rwd_comp = RewardComputer(approach_direction, reward_type, reward_parameters, docking_port_locations, docking_settings)

state1 = np.array([0,-20,0,0,1,0])
state2 = np.array([-2,-2,0,0,1,0])
action1 = np.array([1,1,1])
action2 = np.array([0,0,0])

rwd1 = rwd_comp.get_reward(state1, action1)
print(rwd1)

rwd2 = rwd_comp.get_reward(state2, action2)
print(rwd2)
