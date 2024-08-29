# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "sensitivity_analysis/"
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
    "max_offdir_vel" : 0.1,
    "ideal_dir_vel" : 0.2,
    "ideal_docking_vel" : 0.1,
    "min_dir_vel" : 0.1,
    "max_dir_vel" : 0.2  
}

reward_parameters = {
    "offir_pos_w" : 0.5,
    "dir_pos_w" : 1.0,
    "offdir_rate_w" : 0.1,
    "dir_rate_w" : 2.0,
    "dir_rate_s" : 1.0,
    "conditional_reward_w" : 2.0,
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
            "max_episodes":2500,             # number of simulations to run
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

# main settings
save_dir = main_dir+sub_dir+"main/"
if (not os.path.isdir(save_dir)):
    os.mkdir(save_dir)

with open(save_dir+'main_settings.txt', 'w') as convert_file: 
    convert_file.write(json.dumps(settings))

trainer = Trainer(settings, save_dir)
trainer.start_training()

def run_parameter_values(values, par_name):
    counter = 0
    for val in values:
        counter += 1
        settings_dupe = settings.copy()
        if par_name == "lr":
            settings_dupe[par_name+"_actor"] = val
            settings_dupe[par_name+"_critic"] = val

        elif par_name == "policy_noise":
            settings_dupe["policy_noise"] = val
            settings_dupe["noise_clip"] = 2*val

        else:
            settings_dupe[par_name] = val

        save_dir = main_dir+sub_dir+par_name+f"_{counter}/"
        if (not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        trainer = Trainer(settings_dupe, save_dir)
        trainer.start_training()

        with open(save_dir+'settings.txt', 'w') as convert_file: 
            convert_file.write(json.dumps(settings_dupe))


""" # iterations
values = [10, 1000]
par_name = "n_iters"
run_parameter_values(values, par_name)

# batch size
values = [100, 10000]
par_name = "batch_size"
run_parameter_values(values, par_name)

# learning rate
values = [10**(-4), 10**(-2)]
par_name = "lr"
run_parameter_values(values, par_name)

# gamma
values = [0.90, 0.95, 0.999]
par_name = "gamma"
run_parameter_values(values, par_name)

# exploration_noise
values = [0.1, 0.5]
par_name = "exploration_noise"
run_parameter_values(values, par_name)

# policy_noise
values = [0.1, 0.5]
par_name = "policy_noise"
run_parameter_values(values, par_name)

# polyak
values = [0.99, 0.999]
par_name = "polyak"
run_parameter_values(values, par_name)
 """
# policy delay
values = [1, 3]
par_name = "policy_delay"
run_parameter_values(values, par_name)
