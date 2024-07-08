# Module imports
from sim_utils import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt

# General imports
import numpy as np
import os
import time

#altitude = 1000E3 # meter
altitude = 450E3 # meter
target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
sim_settings = SimSettings(target_kepler_orbit)

initial_cart_state = sim_settings.get_randomized_chaser_state()
prop = sim_settings.setup_simulation(initial_cart_state)

# Create simulation object and propagate dynamics.
t1 = time.process_time()
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)
t2 = time.process_time()

print("Time = ", t2-t1)

# Retrieve all data produced by simulation
propagation_results = dynamics_simulator.propagation_results

# Extract numerical solution for states and dependent variables
state_history = propagation_results.state_history
dependent_variables = propagation_results.dependent_variable_history

###########################################################################
# SAVE RESULTS ############################################################
###########################################################################

save2txt(solution=state_history,
         filename='PropagationHistory.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='PropagationHistory_DependentVariables.dat',
         directory='./'
         )


# Automatically run the post processing script:
with open("plot_trajectory.py") as file:
    exec(file.read())