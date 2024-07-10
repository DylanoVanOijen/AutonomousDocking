# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "hyperparameter_search/"

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
    "lamda" : 1,  # 
    "mu" : 0.05,
    "corridor_penalty" : 1000,
    "far_away_penalty" : 5000,
    "docking_pos_bonus" : 1000,
    "docking_vel_bonus" : 1000,
    "docking_pos_bonus_scaling" : 1000,
    "docking_vel_bonus_scaling" : 1000,
}

settings = {"random_seed":42,
            "max_action":1,
            "gamma": 0.99,
            "batch_size": 100,              # num of transitions sampled from replay buffer
            "lr_actor":0.00001,              # learning rate of actor = alpha
            "lr_critic":0.00001,             # learning rate of critic = beta
            "exploration_noise":0.2, 
            "polyak":0.995,                 # target policy update parameter (1-tau)
            "policy_noise":0.1,             # target policy smoothing noise
            "noise_clip":0.2,
            "policy_delay":2,               # delayed policy updates parameter
            "max_episodes":100,             # number of simulations to run
            "n_iters":100,                   # Number of training iterations per episode
            "fc1_dim":256,                  # Number of nodes in fully connected linear layer 1
            "fc2_dim":256,                  # Number of nodes in fully connected linear layer 2
            "save_each_episode":False,        # Flag to save the models after each epoch instead of only when the results improved
            "approach_direction":"pos_R-bar",# choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
            "reward_type":"simple",          # choose from simple, full or ...
            "action_space_size":3,          # for each direction, pos, neg or no thrust
            "observation_space_size":6,     # pos and vel in TNW frame of Target
            "docking_port_locations":docking_port_locations,
            "docking_settings":docking_settings,
            "reward_parameters":reward_parameters
            }


pars_to_loop = {"nn_dim" : [64, 128, 256, 512],
                "batch_size" : [50, 100, 200, 500],
                "n_iters" : [50, 100, 250],
                "lr" : [10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3)],
                "gamma": [0.9, 0.95, 0.99],
                "policy_delay": [2,3,4],
                "exploration_noise" : [0.05, 0.1, 0.2, 0.3],
                "polyak" : [0.9, 0.95, 0.99, 0.995, 0.999],
                "policy_noise": [0.05, 0.1, 0.2, 0.3],             
                "noise_clip": [0.1, 0.2, 0.3, 0.5],
                }

for par in pars_to_loop.keys():
    values = pars_to_loop[par]

    best_rwd = None
    val_best_rwd = None

    counter = 0

    for val in values:
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
        trainer = Trainer(settings,save_dir)
        trainer.start_training()

        if counter == 0:
            best_rwd = trainer.best_reward
            val_best_rwd = val

        else:
            if trainer.best_reward > best_rwd:
                best_rwd = trainer.best_reward
                val_best_rwd = val

        counter += 1

    print(f"Setting {par} to {val}")

    if par == "nn_dim":
        settings["fc1_dim"] = val_best_rwd
        settings["fc2_dim"] = val_best_rwd

    elif par == "lr":
        settings["lr_actor"] = val_best_rwd
        settings["lr_critic"] = val_best_rwd

    else:
        settings[par] = val_best_rwd

# ND array not serializable, so must convert to lists for storage
for port_name in docking_port_locations:
    port_loc_list = docking_port_locations[port_name].tolist()
    docking_port_locations[port_name] = port_loc_list

with open(main_dir+sub_dir+'settings.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(settings))
