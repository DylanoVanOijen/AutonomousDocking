# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "hyperparameter_search/"

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([-2, -2 , 0])
        }

sim_settings = {
    "step_size": 1.0,
    "orbit_height" : 100000e3   # m
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

settings = {"random_seed":42,
            "max_action":1,
            "gamma": 0.99,
            "buffer_size": 3e3,
            "batch_size": 50,              # num of transitions sampled from replay buffer
            "lr_actor":10**(-6),              # learning rate of actor = alpha
            "lr_critic":10**(-6),             # learning rate of critic = beta
            "exploration_noise":0.01, 
            "polyak":0.995,                 # target policy update parameter (1-tau)
            "policy_noise":0.01,             # target policy smoothing noise
            "noise_clip":0.2,
            "policy_delay":2,               # delayed policy updates parameter
            "max_episodes":500,             # number of simulations to run
            "n_iters":500,                   # Number of training iterations per episode (not used anymore)
            "fc1_dim":128,                  # Number of nodes in fully connected linear layer 1
            "fc2_dim":128,                  # Number of nodes in fully connected linear layer 2
            "save_each_episode":False,        # Flag to save the models after each epoch instead of only when the results improved
            "approach_direction":"pos_R-bar",# choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
            "reward_type":"simple",          # choose from simple, full or ...
            "action_space_size":3,          # for each direction, pos, neg or no thrust
            "observation_space_size":6,     # pos and vel in TNW frame of Target
            "docking_port_locations":docking_port_locations,
            "docking_settings":docking_settings,
            "reward_parameters":reward_parameters,
            "sim_settings":sim_settings
            }


pars_to_loop = {"nn_dim" : [64, 128, 256, 512],
                "batch_size" : [50, 100, 200, 500],
                "lr" : [10**(-7), 10**(-6), 10**(-5), 10**(-4)],
                "policy_delay": [2,3,4],
                "exploration_noise" : [0.01, 0.05, 0.1, 0.2],
                "polyak" : [0.9, 0.95, 0.99, 0.995, 0.999],
                "policy_noise": [0.01, 0.05, 0.1, 0.2],             
                "noise_clip": [0.1, 0.2, 0.3, 0.5],
                }


seeds = [41,42,43]
for par in pars_to_loop.keys():
    values = pars_to_loop[par]

    best_mean_rwd = -np.inf
    val_best_mean_rwd = None

    for val in values:
        mean_higest_rwd = 0
        
        if par == "nn_dim":
            settings["fc1_dim"] = val
            settings["fc2_dim"] = val

        elif par == "lr":
            settings["lr_actor"] = val
            settings["lr_critic"] = val

        else:
            settings[par] = val

        save_dir = main_dir+sub_dir+par+"_"+str(val)+"/"
        if (not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        for seed in seeds:
            settings["random_seed"] = seed
            trainer = Trainer(settings,save_dir)
            trainer.start_training()

            mean_higest_rwd += trainer.best_reward

        mean_higest_rwd /= len(seeds)

        if mean_higest_rwd > best_mean_rwd:
            best_mean_rwd = mean_higest_rwd
            val_best_mean_rwd = val
    
    if par == "nn_dim":
        settings["fc1_dim"] = val_best_mean_rwd
        settings["fc2_dim"] = val_best_mean_rwd

    elif par == "lr":
        settings["lr_actor"] = val_best_mean_rwd
        settings["lr_critic"] = val_best_mean_rwd

    else:
        settings[par] = val_best_mean_rwd

# ND array not serializable, so must convert to lists for storage
for port_name in docking_port_locations:
    port_loc_list = docking_port_locations[port_name].tolist()
    docking_port_locations[port_name] = port_loc_list

with open(main_dir+sub_dir+'settings.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(settings))
