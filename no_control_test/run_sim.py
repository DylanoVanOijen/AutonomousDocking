import numpy as np
from sim_utils import *
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

sim_settings = {
    "step_size": 1.0,
    "orbit_height" : 100000e6   # m
}

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(12,8))


# Sim settings
altitude = sim_settings["orbit_height"]
target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
sim_settings_obj = SimSettings(target_kepler_orbit)

initial_cartesian_state = sim_settings_obj.get_randomized_chaser_state()
prop = sim_settings_obj.setup_simulation(initial_cartesian_state)

# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings_obj.bodies, prop)

states = dynamics_simulator.state_history
dep_vars = dynamics_simulator.dependent_variable_history

states_array = result2array(states)
dep_vars_array = result2array(dep_vars)

#plt.tight_layout()
ax1 = plot_thrust_TNW_frame(ax1, dep_vars_array)
ax2 = plot_trajectory_2d(ax2, states_array, dep_vars_array)
ax4 = plot_velocity_2d(ax4, states_array, dep_vars_array)
ax3 = plot_thrust_body_frame(ax3, dep_vars_array)
#plt.draw()
plt.show()