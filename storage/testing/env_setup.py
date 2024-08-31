# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.data import save2txt

# General imports
import numpy as np
import os

from tudatpy.interface import spice
spice.load_standard_kernels()


simulation_start_epoch = 0.0  # s
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'
max_simtime = 10.0*60.0            # 10 minutes
bodies_to_propagate = ['Chaser']
central_bodies = ['Earth']
integrator_stepsize = 0.1
propagator = "Encke"

bodies_to_create = ['Earth']

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

#body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(self.global_frame_orientation, 'IAU_Earth', 'IAU_Earth', self.simulation_start_epoch)
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.create_empty_body('Chaser')
bodies.get_body('Chaser').set_constant_mass(10e3) 
#bodies.create_empty_body('Target')
#bodies.get_body('Target').set_constant_mass(450e3) 

acceleration_settings_on_vehicle = {'Earth': [propagation_setup.acceleration.point_mass_gravity()]}

# Create acceleration models.
acceleration_settings = {'Chaser': acceleration_settings_on_vehicle}
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

integrator_settings = propagation_setup.integrator.runge_kutta_4(integrator_stepsize)
dependent_variables_to_save = [propagation_setup.dependent_variable.altitude('Chaser', 'Earth')]

altitude = 1000E3 # meter
kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
initial_state = element_conversion.keplerian_to_cartesian(kepler_orbit, earth_gravitational_parameter)

time_termination_settings = propagation_setup.propagator.time_termination(
    simulation_start_epoch + max_simtime,
    terminate_exactly_on_final_condition=False
)

termination_settings_list = [time_termination_settings]

# Create hybrid termination settings object that terminates when one of multiple conditions are met
termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                            fulfill_single_condition=True)

propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                acceleration_models,
                                                                bodies_to_propagate,
                                                                initial_state,
                                                                simulation_start_epoch,
                                                                integrator_settings,
                                                                termination_settings,
                                                                propagator)



# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings)

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