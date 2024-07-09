# Module imports
from training_utils import *

# General imports
import json 

settings = {"random_seed":0,
            "max_action":1,
            "gamma": 0.99,
            "batch_size": 100,              # num of transitions sampled from replay buffer
            "lr_actor":0.001,              # learning rate of actor = alpha
            "lr_critic":0.001,             # learning rate of critic = beta
            "exploration_noise":0.1, 
            "polyak":0.995,                 # target policy update parameter (1-tau)
            "policy_noise":0.1,             # target policy smoothing noise
            "noise_clip":0.2,
            "policy_delay":2,               # delayed policy updates parameter
            "max_episodes":100,             # number of simulations to run
            "n_iters":50,                   # Number of training iterations per episode
            "fc1_dim":512,                  # Number of nodes in fully connected linear layer 1
            "fc2_dim":256,                  # Number of nodes in fully connected linear layer 2
            "save_each_episode":False,        # Flag to save the models after each epoch instead of only when the results improved
            "approach_direction":"pos_R-bar",# choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
            "reward_type":"simple",          # choose from simple, full or ...
            "action_space_size":3,          # for each direction, pos, neg or no thrust
            "observation_space_size":6      # pos and vel in TNW frame of Target
            }

save_folder = "./tmp/td3/"
with open(save_folder+'settings.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(settings))

trainer = Trainer(settings)
trainer.start_training()