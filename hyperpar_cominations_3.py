# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "hyperparameter_combinations_3_"
pre_dir = main_dir+sub_dir

if (not os.path.isdir(pre_dir)):
    os.mkdir(pre_dir)

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": [0, 0, 0]
        }

sim_settings = {
    "step_size": 0.2,
    "orbit_height" : 400e3   # m
}

docking_settings = {
    "KOS_size" : 50,
    "corridor_base_radius": 1,   # meter
    "corridor_angle" : np.deg2rad(25), 
    "max_distance" : 12,
    "max_offdir_pos" : 0.2,
    "max_offdir_vel" : 0.01,
    "ideal_dir_vel" : 0.2,
    "ideal_docking_vel" : 0.1,
    "min_dir_vel" : 0.1,
    "max_dir_vel" : 0.2  
}


reward_parameters = {
    "offir_pos_w" : 0.5,
    "dir_pos_w" : 1.0,
    "offdir_rate_w" : 0.1,
    "dir_rate_w" : 1.0,
    "dir_rate_s" : 1.0,
    "conditinal_reward_w" : 2.0,
    "corridor_penalty" : 100,
    "far_away_penalty" : 100,
    "docking_pos_bonus" : 500,
    "docking_vel_bonus" : 200,
}

settings = {"random_seed":42,
            "max_action":1,
            "gamma": 0.99,
            "buffer_size": 10000000,
            "batch_size": 1000,              # num of transitions sampled from replay buffer
            "lr_actor":10**(-3),              # learning rate of actor = alpha
            "lr_critic":10**(-3),             # learning rate of critic = beta
            "exploration_noise":0.2, 
            "polyak":0.995,                 # target policy update parameter (1-tau)
            "policy_noise":0.2,             # target policy smoothing noise
            "noise_clip":0.4,
            "policy_delay":2,               # delayed policy updates parameter
            "max_episodes":1000,             # number of simulations to run
            "n_iters":100,                   # Number of training iterations per episode (not used anymore)
            "fc1_dim":400,                  # Number of nodes in fully connected linear layer 1
            "fc2_dim":300,                  # Number of nodes in fully connected linear layer 2
            "save_each_episode":False,        # Flag to save the models after each epoch instead of only when the results improved
            "approach_direction":"pos_R-bar",# choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
            "reward_type":"full",          # choose from simple, full or ...
            "action_space_size":3,          # for each direction, pos, neg or no thrust
            "observation_space_size":12,     # pos and vel in TNW frame of Target
            "show_plots":False,
            "docking_port_locations":docking_port_locations,
            "docking_settings":docking_settings,
            "reward_parameters":reward_parameters,
            "sim_settings":sim_settings
            }

pars_to_loop = {"dir_rate_s" : [0.5, 1.0, 2.0],
                "dir_rate_w" : [0.5, 1.0, 2.0],
                "conditional_reward_w" : [1, 2, 5]
                }

start = 1
skiplist = []
option_counter = 0
n_options = len(pars_to_loop["dir_rate_s"])*len(pars_to_loop["dir_rate_w"])*len(pars_to_loop["conditinal_reward_w"])
for a in pars_to_loop["dir_rate_s"]:
    settings["reward_parameters"]["dir_rate_s"] = a

    for b in pars_to_loop["dir_rate_w"]:
        settings["reward_parameters"]["dir_rate_w"] = b

        for c in pars_to_loop["conditinal_reward_w"]:
            settings["reward_parameters"]["conditinal_reward_w"] = c

            option_counter += 1
            print(f"Running option {option_counter} out of {n_options}")
            save_dir = main_dir+sub_dir+f"option_{option_counter}/"
            if (not os.path.isdir(save_dir)):
                os.mkdir(save_dir)

            if option_counter >= start and option_counter not in skiplist:
                trainer = Trainer(settings,save_dir)
                trainer.start_training()

                with open(save_dir+'settings.txt', 'w') as convert_file: 
                    convert_file.write(json.dumps(settings))
