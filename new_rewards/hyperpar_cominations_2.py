# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "hyperparameter_combinations_2_"
pre_dir = main_dir+sub_dir

if (not os.path.isdir(pre_dir)):
    os.mkdir(pre_dir)

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([0, 0, 0])
        }

sim_settings = {
    "step_size": 1.0,
    "orbit_height" : 100000e3   # m
}

docking_settings = {
    "KOS_size" : 50,
    "corridor_base_radius": 1,   # meter
    "corridor_angle" : np.deg2rad(25), 
    "max_distance" : 12,
    "max_offdir_pos" : 0.2,
    "max_offdir_vel" : 0.01,
    "ideal_dir_vel" : 0.2,
    "min_dir_vel" : 0.1,
    "max_dir_vel" : 0.2  
}


reward_parameters = {
    "offir_pos_w" : 0.5,
    "dir_pos_w" : 0.5,
    "offdir_rate_w" : 0.1,
    "dir_rate_w" : 1.0,
    "dir_rate_s" : 5.0,
    "conditinal_reward_w" : 2.0,
    "corridor_penalty" : 100,
    "far_away_penalty" : 100,
    "docking_pos_bonus" : 200,
    "docking_vel_bonus" : 200,
}

settings = {"random_seed":42,
            "max_action":1,
            "gamma": 0.999,
            "buffer_size": 10000000,
            "batch_size": 1000,              # num of transitions sampled from replay buffer
            "lr_actor":10**(-3),              # learning rate of actor = alpha
            "lr_critic":10**(-3),             # learning rate of critic = beta
            "exploration_noise":0.1, 
            "polyak":0.995,                 # target policy update parameter (1-tau)
            "policy_noise":0.1,             # target policy smoothing noise
            "noise_clip":0.2,
            "policy_delay":2,               # delayed policy updates parameter
            "max_episodes":500,             # number of simulations to run
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

pars_to_loop = {"dir_rate_s" : [1, 5, 10],
                "dir_rate_w" : [0.1, 1.0, 2.0],
                "off_dir_rate_w" : [0.1, 0.5, 1.0],
                "dir_pos_w" : [0.1, 0.5, 1.0]
                }

start = 1
skiplist = []
seeds = [42]
option_counter = 0
n_options = len(pars_to_loop["dir_rate_s"])*len(pars_to_loop["dir_rate_w"])*len(pars_to_loop["off_dir_rate_w"])*len(pars_to_loop["dir_pos_w"])
for a in pars_to_loop["dir_rate_s"]:
    settings["reward_parameters"]["dir_rate_s"] = a

    for b in pars_to_loop["dir_rate_w"]:
        settings["reward_parameters"]["dir_rate_w"]

        for c in pars_to_loop["off_dir_rate_w"]:
            settings["reward_parameters"]["off_dir_rate_w"] = c

            for d in pars_to_loop["dir_pos_w"]:
                settings["reward_parameters"]["dir_pos_w"] = d

                option_counter += 1
                print(f"Running option {option_counter} out of {n_options}")
                save_dir = main_dir+sub_dir+f"option_{option_counter}/"
                if (not os.path.isdir(save_dir)):
                    os.mkdir(save_dir)

                if option_counter >= start and option_counter not in skiplist:

                    for seed in seeds:
                        settings["random_seed"] = seed
                        seed_dir = save_dir + f"/seed_{seed}/"
                        if (not os.path.isdir(seed_dir)):
                            os.mkdir(seed_dir)


                        trainer = Trainer(settings,seed_dir)
                        trainer.start_training()

                        if option_counter == start and seed == seeds[0]:
                            # ND array not serializable, so must convert to lists for storage
                            for port_name in docking_port_locations:
                                port_loc_list = docking_port_locations[port_name].tolist()
                                docking_port_locations[port_name] = port_loc_list

                        with open(seed_dir+'settings.txt', 'w') as convert_file: 
                            convert_file.write(json.dumps(settings))
