
# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
import tudatpy.util as util

# Problem-specific imports
import CapsuleEntryUtilities as Util
from CapsuleEntryProblem import ShapeOptimizationProblem

spice_interface.load_standard_kernels()

current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
# Set vehicle properties
capsule_density = 250.0  # kg m-3





#######################################################################
### TERMINATION SETTINGS AND RETRIEVING DEPENDENT VARIABLES TO SAVE ###
#######################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)


parameters = dict()
objectives_and_constraints = dict()

for simulation_index in range(number_of_simulations):

    print('Simulation', simulation_index)

    # If Monte Carlo, a random value is chosen with a uniform distribtion (NOTE: You can change the distribution)
    for parameter_index in range(number_of_parameters):
        shape_parameters[parameter_index] = np.random.uniform(decision_variable_range[0][parameter_index], decision_variable_range[1][parameter_index])

    print('Parameters:', shape_parameters)
        
    parameters[simulation_index] = shape_parameters.copy()

    # Problem class is created
    current_capsule_entry_problem = ShapeOptimizationProblem(bodies,
                                                     termination_settings,
                                                     capsule_density,
                                                     simulation_start_epoch,
                                                     decision_variable_range)


    # NOTE: Propagator settings, termination settings, and initial_propagation_time are defined in the fitness function
    fitness = current_capsule_entry_problem.fitness(shape_parameters) # RIGHT NOW, FITNESS IS ALWAYS 0.0! MODIFY THE FUNCTION APPROPRIATELY
    print('Fitness:', fitness)
    
    objectives_and_constraints[simulation_index] = fitness

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = current_capsule_entry_problem.get_last_run_dynamics_simulator().state_history
    dependent_variable_history = current_capsule_entry_problem.get_last_run_dynamics_simulator().dependent_variable_history

    # Get output path
    subdirectory = '/DesignSpace_%s/Run_%s'%(design_space_method, simulation_index)

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

    # Delete data in aerodynamic coefficient database
    bodies.get_body( 'Capsule' ).aerodynamic_coefficient_interface.clear_data()

if write_results_to_file:
    subdirectory = '/DesignSpace_%s'%(design_space_method)
    output_path = current_dir + subdirectory
    save2txt(parameters, 'parameter_values.dat', output_path)
    save2txt(objectives_and_constraints, 'objectives_constraints.dat', output_path)
