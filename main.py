# Module imports
from SimUtils import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt

# General imports
import numpy as np
import os


sim_settings = SimSettings

prop = sim_settings.setup_simulation()

# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)

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