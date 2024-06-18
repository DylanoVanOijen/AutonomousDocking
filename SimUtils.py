# Tudat imports
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.astro import element_conversion

# Tudat SPICE kernel setup
from tudatpy.interface import spice
spice.load_standard_kernels()

# General mports
import numpy as np
import math

class SimSettings:
    def __init__(self, target_kepler_orbit):
        # Vehicle properties
        self.isp = 5e3
        self.chaser_mass = 10e3
        self.target_mass = 450e3

        self.target_kepler_orbit = target_kepler_orbit
        self.simulation_start_epoch = 0.0  # s
        self.global_frame_origin = 'Earth'
        self.global_frame_orientation = 'J2000'
        self.max_simtime = 10.0*60.0            # 10 minutes
        #self.max_simtime = 10
        self.bodies_to_propagate = ['Target', 'Chaser']
        self.central_bodies = ['Earth', 'Earth']
        self.integrator_stepsize = 0.1
        self.propagator = propagation_setup.propagator.encke
        self.chaser_GNC = ChaserGNC() 
        self.bodies = self.get_environment_settings()
        self.acceleration_models = self.get_acceleration_settings()
        self.integrator_settings = self.get_integrator_settings()
        self.dep_vars_to_save = self.get_dependent_variables_to_save()
        self.termination_settings = self.get_termination_settings()
        self.target_cartesian_orbit = self.get_cart_state(self.target_kepler_orbit)

    def get_environment_settings(self):
        bodies_to_create = ['Earth']

        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            self.global_frame_origin,
            self.global_frame_orientation)

        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(self.global_frame_orientation, 'IAU_Earth', 'IAU_Earth', self.simulation_start_epoch)
        bodies = environment_setup.create_system_of_bodies(body_settings)

        self.add_target_body(bodies)
        self.add_chaser_body(bodies)

        return bodies
    
    def add_chaser_body(self, bodies):
        bodies.create_empty_body('Chaser')
        bodies.get_body('Chaser').set_constant_mass(self.chaser_mass) 

        rotation_model_settings = environment_setup.rotation_model.orbital_state_direction_based(self.central_bodies[1], is_colinear_with_velocity=True, 
                                                                                                 direction_is_opposite_to_vector=False, base_frame = "",
                                                                                                 target_frame = "ChaserFixed" )
        environment_setup.add_rotation_model(bodies, 'Chaser', rotation_model_settings )

        thrust_magnitude_settings_XP = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Xp, specific_impulse=self.isp)
        thrust_magnitude_settings_Xm = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Xm, specific_impulse=self.isp)
        thrust_magnitude_settings_YP = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Yp, specific_impulse=self.isp)
        thrust_magnitude_settings_Ym = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Ym, specific_impulse=self.isp)
        thrust_magnitude_settings_ZP = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Zp, specific_impulse=self.isp)
        thrust_magnitude_settings_Zm = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Zm, specific_impulse=self.isp)

        environment_setup.add_engine_model('Chaser', 'X+', thrust_magnitude_settings_XP, bodies, np.array([1,0,0]))
        environment_setup.add_engine_model('Chaser', 'X-', thrust_magnitude_settings_Xm, bodies, np.array([-1,0,0]))
        environment_setup.add_engine_model('Chaser', 'Y+', thrust_magnitude_settings_YP, bodies, np.array([0,1,0]))
        environment_setup.add_engine_model('Chaser', 'Y-', thrust_magnitude_settings_Ym, bodies, np.array([0,-1,0]))
        environment_setup.add_engine_model('Chaser', 'Z+', thrust_magnitude_settings_ZP, bodies, np.array([0,0,1]))
        environment_setup.add_engine_model('Chaser', 'Z-', thrust_magnitude_settings_Zm, bodies, np.array([0,0,-1]))

    
    def add_target_body(self, bodies):
        bodies.create_empty_body('Target')
        bodies.get_body('Target').set_constant_mass(self.target_mass) 
        #environment_setup.add_rotation_model( bodies, 'Capsule',
        #                                    environment_setup.rotation_model.aerodynamic_angle_based(
        #                                        'Earth', 'J2000', 'CapsuleFixed', angle_function ))


   
    def get_acceleration_settings(self):
        acceleration_settings_on_vehicle = {'Earth': [propagation_setup.acceleration.point_mass_gravity()],
                                            'Chaser':[propagation_setup.acceleration.thrust_from_all_engines()]}

        # Create acceleration models.
        acceleration_settings = {'Chaser': acceleration_settings_on_vehicle, 'Target': acceleration_settings_on_vehicle}
        acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies,
            acceleration_settings,
            self.bodies_to_propagate,
            self.central_bodies)

        return acceleration_models
    
    def get_integrator_settings(self):
        # Create numerical integrator settings.
        integrator_settings = propagation_setup.integrator.runge_kutta_4(self.integrator_stepsize)
        return integrator_settings
    
    def get_dependent_variables_to_save(self):
        dependent_variables_to_save = [propagation_setup.dependent_variable.tnw_to_inertial_rotation_matrix('Target', 'Earth')                  
                                       ]
        return dependent_variables_to_save
    
    def setup_simulation(self, initial_state):
        propagator_settings = propagation_setup.propagator.translational(central_bodies = self.central_bodies,
                                                                        acceleration_models = self.acceleration_models,
                                                                        bodies_to_integrate = self.bodies_to_propagate,
                                                                        initial_states = np.concatenate((self.target_cartesian_orbit, initial_state)),
                                                                        initial_time = self.simulation_start_epoch,
                                                                        integrator_settings = self.integrator_settings,
                                                                        termination_settings = self.termination_settings,
                                                                        propagator = self.propagator,
                                                                        output_variables = self.dep_vars_to_save)
        
        return propagator_settings
    

    # combine the different termination options
    def get_termination_settings(self):

        time_termination_settings = propagation_setup.propagator.time_termination(
            self.simulation_start_epoch + self.max_simtime,
            terminate_exactly_on_final_condition=False
        )

        # Altitude
        #lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        #    dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
        #    limit_value=termination_altitude,
        #    use_as_lower_limit=True,
        #    terminate_exactly_on_final_condition=False
        #)
        # Define list of termination settings
        termination_settings_list = [time_termination_settings]

        # Create hybrid termination settings object that terminates when one of multiple conditions are met
        hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                    fulfill_single_condition=True)
        return hybrid_termination_settings
    
    def get_cart_state(self, kepler_state):
        earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter
        cart_state = element_conversion.keplerian_to_cartesian(kepler_state, earth_gravitational_parameter)
        return cart_state
    
    # Returns randomized cartesian state
    def get_randomized_chaser_state(self):
        randomized_state = np.copy(self.target_cartesian_orbit)
        randomized_state[1] += 10            
        return randomized_state
    

class ChaserGNC:
    def __init__(self, bodies: environment.SystemOfBodies):
        # Extract the STS and Earth bodies
        self.vehicle = bodies.get_body("Chaser")
        self.earth = bodies.get_body("Earth")

        self.current_time = float("NaN")

    def get_aerodynamic_angles(self, current_time: float):

        # Update the class to the current time
        self.update_guidance( current_time )
        
        # Return angles calculated by update function
        return np.array([self.angle_of_attack, 0.0, self.bank_angle])


    def get_thrust_magnitude_Xp(self, current_time: float):
        self.update_guidance( current_time )
        return self.thrust_magnitude_Xp
    
    def get_thrust_magnitude_Xm(self, current_time: float):
        self.update_guidance( current_time )
        return self.thrust_magnitude_Xm
    
    def get_thrust_magnitude_Yp(self, current_time: float):
        self.update_guidance( current_time )
        return self.thrust_magnitude_Yp
    
    def get_thrust_magnitude_Ym(self, current_time: float):
        self.update_guidance( current_time )
        return self.thrust_magnitude_Ym
    
    def get_thrust_magnitude_Zp(self, current_time: float):
        self.update_guidance( current_time )
        return self.thrust_magnitude_Zp
    
    def get_thrust_magnitude_Zm(self, current_time: float):
        self.update_guidance( current_time )
        return self.thrust_magnitude_Zm


    def update_GNC(self, current_time: float):

        if( math.isnan( current_time ) ):
	    # Set the model's current time to NaN, indicating that it needs to be updated 
            self.current_time = float("NaN")
        elif( current_time != self.current_time ):

            # Calculate current body orientation through angle of attack and bank angle
            self.angle_of_attack = ...
            self.bank_angle = ...

            # Calculate current thrust magnitude
            self.thrust_magnitude = ...

	    # Set the model's current time, indicating that it has been updated
            self.current_time = current_time