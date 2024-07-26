# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "hyperparameter_combinations/"
pre_dir = main_dir+sub_dir

if (not os.path.isdir(pre_dir)):
    os.mkdir(pre_dir)

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([-2, -2 , 0])
        }

sim_settings = {
    "step_size": 1.0,
    "orbit_height" : 100000e3   # m
}

docking_settings = {
    "KOS_size" : 50,
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
    "lamda" : 0.1,  # 
    "mu" : 0.05,
    "corridor_penalty" : 10,
    "far_away_penalty" : 50,
    "docking_pos_bonus" : 10,
    "docking_vel_bonus" : 10,
    "docking_pos_bonus_scaling" : 10,
    "docking_vel_bonus_scaling" : 10,
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
            "max_episodes":250,             # number of simulations to run
            "n_iters":500,                   # Number of training iterations per episode (not used anymore)
            "fc1_dim":128,                  # Number of nodes in fully connected linear layer 1
            "fc2_dim":128,                  # Number of nodes in fully connected linear layer 2
            "save_each_episode":False,        # Flag to save the models after each epoch instead of only when the results improved
            "approach_direction":"pos_R-bar",# choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
            "reward_type":"full",          # choose from simple, full or ...
            "action_space_size":3,          # for each direction, pos, neg or no thrust
            "observation_space_size":6,     # pos and vel in TNW frame of Target
            "docking_port_locations":docking_port_locations,
            "docking_settings":docking_settings,
            "reward_parameters":reward_parameters,
            "sim_settings":sim_settings
            }
transitions = 60
pars_to_loop = {"buffer_size" : [20*transitions, 50*transitions, 150*transitions],
                "batch_size" : [2*transitions, 5*transitions, 10*transitions],
                "lr" : [10**(-7), 10**(-6), 10**(-5)],
                "n_iters": [10,100,1000]
                }


seeds = [42]

option_counter = 0
n_options = len(pars_to_loop["buffer_size"])*len(pars_to_loop["batch_size"])*len(pars_to_loop["lr"])*len(pars_to_loop["n_iters"])
for buff_size in pars_to_loop["buffer_size"]:
    settings["buffer_size"] = buff_size

    for batch_size in pars_to_loop["batch_size"]:
        settings["batch_size"] = batch_size

        for lr in pars_to_loop["lr"]:
            settings["lr_actor"] = lr
            settings["lr_critic"] = lr

            for n_iters in pars_to_loop["n_iters"]:
                settings["n_iters"] = n_iters
                option_counter += 1
                print(f"Running option {option_counter} out of {n_options}")

                save_dir = main_dir+sub_dir+f"option_{option_counter}/"
                if (not os.path.isdir(save_dir)):
                    os.mkdir(save_dir)

                for seed in seeds:
                    settings["random_seed"] = seed
                    seed_dir = save_dir + f"/seed_{seed}/"
                    if (not os.path.isdir(seed_dir)):
                        os.mkdir(seed_dir)

                    trainer = Trainer(settings,seed_dir)
                    trainer.start_training()

                    if option_counter == 1 and seed == seeds[0]:
                        # ND array not serializable, so must convert to lists for storage
                        for port_name in docking_port_locations:
                            port_loc_list = docking_port_locations[port_name].tolist()
                            docking_port_locations[port_name] = port_loc_list

                    with open(seed_dir+'settings.txt', 'w') as convert_file: 
                        convert_file.write(json.dumps(settings))
