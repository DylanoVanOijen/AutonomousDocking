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

folder = "./TrainingOutputs/main_test/"
#folder = "./TrainingOutputs/hyperparameter_combinations_option_14/seed_42/"

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


agent.load_models()

arrival_times = []
off_axis_distances = []
off_axis_velocities = []
dir_velocities = []
docking_successes = []

for i in range(n_samples):
    if i % (n_samples // 20) == 0:
        print(f"At {i / n_samples * 100}%")

    initial_cartesian_state = sim_settings.get_randomized_chaser_state(-1)
    prop = sim_settings.setup_simulation(initial_cartesian_state)
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)

    states = dynamics_simulator.state_history
    dep_vars = dynamics_simulator.dependent_variable_history

    states_array = result2array(states)
    dep_vars_array = result2array(dep_vars)

    arrival_time, off_axis_distance, off_axis_velocity, dir_velocity, docking_success = compute_MC_statistics(states_array, dep_vars_array, settings)

    arrival_times.append(arrival_time)
    off_axis_distances.append(off_axis_distance)
    off_axis_velocities.append(off_axis_velocity)
    dir_velocities.append(dir_velocity)
    docking_successes.append(docking_success)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
ax1 = plot_arrival_times(ax1, arrival_times, docking_successes, 50)
ax2 = plot_off_axis_distances(ax2, off_axis_distances, docking_successes, 50)
ax4 = plot_off_axis_velocities(ax4, off_axis_velocities, docking_successes, 50)
ax3 = plot_dir_axis_velocities(ax3, dir_velocities, docking_successes, 50)
fig.savefig("./plots/MC_analysis.png")

sucess_rate = sum(docking_successes)/len(docking_successes)*100
print(f"Option {i}, sucessrate = {sucess_rate:.2f}")