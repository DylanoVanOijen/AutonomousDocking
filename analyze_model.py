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
folder = "./TrainingOutputs/hyperparameter_combinations/option_18/seed_42/"

# Reading the data from the file and converting it back to a dictionary
with open(folder+'settings.txt', 'r') as convert_file:
    settings = json.load(convert_file)

# Convert docking port locations back from list to NDarray
port_locs = settings["docking_port_locations"]
for port_name in port_locs:
    port_loc_array = np.array(port_locs[port_name])
    port_locs[port_name] = port_loc_array
settings["docking_port_locations"] = port_locs
#settings["exploration_noise"] = 0
#settings["noise_clip"] = 0

#torch.manual_seed(settings["random_seed"])
#np.random.seed(settings["random_seed"])  

torch.manual_seed(42)
np.random.seed(42)  

print(settings["lr_actor"], settings["batch_size"])

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

initial_cartesian_state = sim_settings.get_randomized_chaser_state()
prop = sim_settings.setup_simulation(initial_cartesian_state)

agent.load_models()

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)

states = dynamics_simulator.state_history
dep_vars = dynamics_simulator.dependent_variable_history

states_array = result2array(states)
dep_vars_array = result2array(dep_vars)
print(dep_vars_array.shape)

final_reward = agent.episode_reward

print(f"Obtained reward: {final_reward}")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
ax1 = plot_trajectory_2d(ax1, states_array, dep_vars_array)
#ax1.plot(dep_vars_array[:,0], dep_vars_array[:,22], label = "x dep", color = "cyan", linestyle = "dashed")
#ax1.plot(dep_vars_array[:,0], dep_vars_array[:,23], label = "y dep", color = "magenta", linestyle = "dashed")
#ax1.plot(dep_vars_array[:,0], dep_vars_array[:,24], label = "z dep", color = "yellow", linestyle = "dashed")
ax1.legend()
ax2 = plot_velocity_2d(ax2, states_array, dep_vars_array)
#ax2.plot(dep_vars_array[:,0], dep_vars_array[:,25], label = "x dep", color = "cyan", linestyle = "dashed")
#ax2.plot(dep_vars_array[:,0], dep_vars_array[:,26], label = "y dep", color = "magenta", linestyle = "dashed")
#ax2.plot(dep_vars_array[:,0], dep_vars_array[:,27], label = "z dep", color = "yellow", linestyle = "dashed")
ax2.legend()
ax3 = plot_thrust_body_frame(ax3, dep_vars_array)
ax4 = plot_thrust_TNW_frame(ax4, dep_vars_array)
fig.tight_layout()
fig.savefig("./plots/model_analysis.png")



port_loc = settings["docking_port_locations"]["pos_R-bar"]
fig2 = plt.figure()
ax_3d = fig2.add_subplot(projection='3d')
ax_3d = plot_trajectory_3d(ax_3d, states_array, dep_vars_array, port_loc)
fig2.tight_layout()
fig2.savefig("./plots/trajectory_3D.png")

plt.show()