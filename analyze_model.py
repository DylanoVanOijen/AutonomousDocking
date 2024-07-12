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

folder = "./TrainingOutputs/main/"

# Reading the data from the file and converting it back to a dictionary
with open(folder+'settings.txt', 'r') as convert_file:
    settings = json.load(convert_file)

# Convert docking port locations back from list to NDarray
port_locs = settings["docking_port_locations"]
for port_name in port_locs:
    port_loc_array = np.array(port_locs[port_name])
    port_locs[port_name] = port_loc_array
settings["docking_port_locations"] = port_locs

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
altitude = 450E3 # meter
target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
sim_settings = SimSettings(target_kepler_orbit, agent)

initial_cartesian_state = sim_settings.get_randomized_chaser_state()
prop = sim_settings.setup_simulation(initial_cartesian_state)

agent.load_models()

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)

states = dynamics_simulator.state_history
dep_vars = dynamics_simulator.dependent_variable_history

states_array = result2array(states)
dep_vars_array = result2array(dep_vars)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
ax1 = plot_trajectory_2d(ax1, states_array, dep_vars_array)
ax2 = plot_velocity_2d(ax2, states_array, dep_vars_array)
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