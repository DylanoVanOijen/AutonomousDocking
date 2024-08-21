from post_process_utils import *
from TD3_agent import *
from sim_utils import *

import json

from sim_utils import *
from TD3_agent import *
from post_process_utils import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array

# General imports
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#folder = "./TrainingOutputs/main_test/"
folder = "./TrainingOutputs/hyperparameter_combinations_option_14/seed_42/"

n_samples = 1000

# Reading the data from the file and converting it back to a dictionary
with open(folder+'settings.txt', 'r') as convert_file:
    settings = json.load(convert_file)

# Convert docking port locations back from list to NDarray
port_locs = settings["docking_port_locations"]
for port_name in port_locs:
    port_loc_array = np.array(port_locs[port_name])
    port_locs[port_name] = port_loc_array
settings["docking_port_locations"] = port_locs
settings["policy_noise"] = 0
settings["exploration_noise"] = 0
settings["noise_clip"] = 0

#torch.manual_seed(settings["random_seed"])
#np.random.seed(settings["random_seed"])  

torch.manual_seed(42)
np.random.seed(42)  

# Creating agent
agent = Agent( alpha=settings["lr_actor"], beta=settings["lr_critic"], 
                        state_dim=settings["observation_space_size"], action_dim=settings["action_space_size"], 
                        fc1_dim=settings["fc1_dim"], fc2_dim=settings["fc2_dim"], 
                        max_action=settings["max_action"], buffer_size=settings["buffer_size"], batch_size=settings["batch_size"], 
                        gamma=settings["gamma"], polyak=settings["polyak"], 
                        policy_noise=settings["policy_noise"], noise_clip=settings["noise_clip"], 
                        policy_delay=settings["policy_delay"], exploration_noise=settings["exploration_noise"], 
                        approach_direction=settings["approach_direction"], 
                        reward_type=settings["reward_type"], reward_parameters=settings["reward_parameters"], 
                        docking_ports=settings["docking_port_locations"], docking_settings=settings["docking_settings"],
                        save_folder=folder)
    

# Sim settings
altitude = settings["sim_settings"]["orbit_height"]
target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
sim_settings = SimSettings(target_kepler_orbit, agent, settings["reward_type"])

initial_cartesian_state = sim_settings.get_randomized_chaser_state(-1)
prop = sim_settings.setup_simulation(initial_cartesian_state)

agent.load_models()

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)

states = dynamics_simulator.state_history
dep_vars = dynamics_simulator.dependent_variable_history

states_array = result2array(states)
dep_vars_array = result2array(dep_vars)

final_reward = agent.episode_reward

for i in range()