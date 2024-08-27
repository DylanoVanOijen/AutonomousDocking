import numpy as np
from TD3_agent import *

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([0, 0, 0])
        }

sim_settings = {
    "step_size": 1.0,
    "orbit_height" : 100000e3   # m
}

docking_settings = {
    "KOS_size" : 50,
    "corridor_base_radius": 0.5,   # meter
    "corridor_angle" : np.deg2rad(20), 
    "max_distance" : 25,
    "max_offdir_pos" : 0.2,
    "max_offdir_vel" : 0.01,
    "ideal_dir_vel" : 0.2,
    "min_dir_vel" : 0.1,
    "max_dir_vel" : 0.2  
}

reward_parameters = {
    "eta" : 10,
    "kappa" : 0.1,  # tanh scaling
    "lamda" : 0.1,  # 
    "mu" : 0.05,
    "corridor_penalty" : 10,
    "far_away_penalty" : 50,
    "docking_pos_bonus" : 50,
    "docking_vel_bonus" : 50,
    "docking_pos_bonus_scaling" : 50/0.5,
    "docking_vel_bonus_scaling" : 50/0.1,
}

approach_direction = "pos_R-bar"    # choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
reward_type = "full"           # choose from simple, full or ...
rwd_comp = RewardComputer(approach_direction, reward_type, reward_parameters, docking_port_locations, docking_settings)

state1 = np.array([0,-10,0,0,0,0])
state2 = np.array([0,-9,0,0,0,0])
action1 = np.array([0,0,0])
action2 = np.array([0,0,0])

rwd1 = rwd_comp.get_reward(state1, action1)
print(rwd1)

rwd2 = rwd_comp.get_reward(state2, action2)
print(rwd2)
