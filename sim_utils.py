# Module imports
from TD3_agent import *

# Tudat imports
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.astro import frame_conversion

# Tudat SPICE kernel setup
from tudatpy.interface import spice
spice.load_standard_kernels()

# General mports
import numpy as np
import math

class SimSettings:
    def __init__(self, target_kepler_orbit, agent):
        # Vehicle properties
        self.isp = 300      # Seconds
        self.thrust = 1000  # Newton
        self.chaser_mass = 10e3
        self.target_mass = 450e3

        self.agent = None
        self.target_kepler_orbit = target_kepler_orbit
        self.simulation_start_epoch = 0.0  # s
        self.global_frame_origin = 'Earth'
        self.global_frame_orientation = 'J2000'
        self.max_simtime = 2.0*60.0            # 5 minutes
        #self.max_simtime = 100
        self.bodies_to_propagate = ['Target', 'Chaser']
        self.central_bodies = ['Earth', 'Earth']
        self.integrator_stepsize = 1.0
        self.propagator = propagation_setup.propagator.encke
        self.chaser_GNC = ChaserGNC(self.thrust, self.integrator_stepsize, agent) 
        self.bodies = self.get_environment_settings()
        self.acceleration_models = self.get_acceleration_settings()
        self.integrator_settings = self.get_integrator_settings()
        self.dep_vars_to_save = self.get_dependent_variables_to_save()
        self.termination_settings = self.get_termination_settings()
        self.target_cartesian_orbit = self.get_cart_state(self.target_kepler_orbit)
        self.chaser_GNC.add_bodies(self.bodies)
        self.observation = np.zeros(6)


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
    
    def add_target_body(self, bodies):
        bodies.create_empty_body('Target')
        bodies.get_body('Target').set_constant_mass(self.target_mass) 
        #environment_setup.add_rotation_model( bodies, 'Capsule',
        #                                    environment_setup.rotation_model.aerodynamic_angle_based(
        #                                        'Earth', 'J2000', 'CapsuleFixed', angle_function ))
    
    def add_chaser_body(self, bodies):
        bodies.create_empty_body('Chaser')
        bodies.get_body('Chaser').set_constant_mass(self.chaser_mass) 

                                                                                                        # check orientation of rot model
        rotation_model_settings = environment_setup.rotation_model.orbital_state_direction_based('Earth', is_colinear_with_velocity=True, 
                                                                                                 direction_is_opposite_to_vector=True, base_frame = self.global_frame_orientation,
                                                                                                 target_frame = "ChaserFixed" )
        environment_setup.add_rotation_model(bodies, 'Chaser', rotation_model_settings )

        thrust_magnitude_settings_Xp = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Xp, specific_impulse=self.isp)
        thrust_magnitude_settings_Yp = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Yp, specific_impulse=self.isp)
        thrust_magnitude_settings_Zp = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(self.chaser_GNC.get_thrust_magnitude_Zp, specific_impulse=self.isp)

        # Need the eps for the while as long as rotational dynamics are not propagated (I think)
        # Because problem: +X and -X direction thrust give same result (maybe quaterion rotation singularity?)
        environment_setup.add_engine_model('Chaser', 'X+', thrust_magnitude_settings_Xp, bodies, np.array([1,-np.finfo(float).eps,0]))
        environment_setup.add_engine_model('Chaser', 'Y+', thrust_magnitude_settings_Yp, bodies, np.array([0,1,0]))
        environment_setup.add_engine_model('Chaser', 'Z+', thrust_magnitude_settings_Zp, bodies, np.array([0,0,1]))

   
    def get_acceleration_settings(self):
        acceleration_settings_on_chaser = {'Earth': [propagation_setup.acceleration.point_mass_gravity()],
                                            'Chaser':[propagation_setup.acceleration.thrust_from_all_engines()]}

        acceleration_settings_on_Target = {'Earth': [propagation_setup.acceleration.point_mass_gravity()]}

        # Create acceleration models.
        acceleration_settings = {'Chaser': acceleration_settings_on_chaser, 'Target': acceleration_settings_on_Target}
        acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies,
            acceleration_settings,
            self.bodies_to_propagate,
            self.central_bodies)

        return acceleration_models
    
    def get_integrator_settings(self):
        # Create numerical integrator settings.
        coeff_set = propagation_setup.integrator.CoefficientSets.rk_4
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(self.integrator_stepsize, coeff_set)
        return integrator_settings
    
    def get_dependent_variables_to_save(self):
        dependent_variables_to_save = [propagation_setup.dependent_variable.tnw_to_inertial_rotation_matrix('Target', 'Earth'),
                                       propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame('Chaser'),
                                       propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.thrust_acceleration_type , 'Chaser', 'Chaser')                  
                                       ]
        return dependent_variables_to_save
    
    def setup_simulation(self, initial_state):
        # reset GNC stuff
        self.chaser_GNC.reset()

        # get propagation settings
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
            terminate_exactly_on_final_condition=True
        )

        outside_cone_termination_settings = propagation_setup.propagator.custom_termination(self.chaser_GNC.agent.reward_computer.outside_cone_interface)
        is_docking_termination_settings = propagation_setup.propagator.custom_termination(self.chaser_GNC.agent.reward_computer.is_docking_interface)
        too_far_termination_settings = propagation_setup.propagator.custom_termination(self.chaser_GNC.agent.reward_computer.too_far_interface)

        # Define list of termination settings
        termination_settings_list = [time_termination_settings, outside_cone_termination_settings, is_docking_termination_settings, too_far_termination_settings]

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
        randomized_state[0] += -10
        #randomized_state[0] += np.random.uniform(-20,-10)
        #randomized_state[1] += np.random.uniform(-5, 5)
        #randomized_state[2] += np.random.uniform(-5, 5)

        #self.observation =                 
        return randomized_state
    

class ChaserGNC:
    def __init__(self, thrust, dt, agent):
        # Extract the STS and Earth bodies
        self.chaser = None
        self.target = None
        self.earth = None
        
        self.agent = agent

        self.dt = dt
        self.max_impulse = thrust*dt    # Newton seconds
        self.rel_tol = dt/10

        self.reset()

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.counter = 0
    
        self.thrust_magnitude_Xp = 0.0
        self.thrust_magnitude_Yp = 0.0
        self.thrust_magnitude_Zp = 0.0

        # start point such that first processed GNC command will occur at t=0
        self.processed_time = -self.dt

    def add_bodies(self, bodies: environment.SystemOfBodies):
        self.chaser = bodies.get_body("Chaser")
        self.target = bodies.get_body("Target")
        self.earth = bodies.get_body("Earth")

        # Extract the Chaser and Target flight conditions, angle calculator, and aerodynamic coefficient interface
        environment_setup.add_flight_conditions(bodies, 'Chaser', 'Earth')
        environment_setup.add_flight_conditions(bodies, 'Target', 'Earth' )
        self.chaser_flight_conditions = bodies.get_body("Chaser").flight_conditions
        self.target_flight_conditions = bodies.get_body("Target").flight_conditions
        self.chaser_aerodynamic_angle_calculator = self.chaser_flight_conditions.aerodynamic_angle_calculator
        self.target_aerodynamic_angle_calculator = self.target_flight_conditions.aerodynamic_angle_calculator

    def get_thrust_magnitude_Xp(self, current_time: float):
        self.update_GNC( current_time )
        return self.thrust_magnitude_Xp
        
    def get_thrust_magnitude_Yp(self, current_time: float):
        self.update_GNC( current_time )
        return self.thrust_magnitude_Yp
        
    def get_thrust_magnitude_Zp(self, current_time: float):
        self.update_GNC( current_time )
        return self.thrust_magnitude_Zp
    

    def update_GNC(self, current_time: float):
        if ( current_time == self.processed_time+self.dt ) :       
            delta_pos_inertial = self.chaser.position-self.target.position
            delta_vel_inertial = self.chaser.velocity-self.target.velocity

            inertial_to_TNW_rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(self.target.state, True)

            chaser_pos_TNW = inertial_to_TNW_rotation_matrix@delta_pos_inertial
            chaser_vel_TNW = inertial_to_TNW_rotation_matrix@delta_vel_inertial

            state = np.concatenate((chaser_pos_TNW, chaser_vel_TNW))
            action = self.agent.compute_action(state)
            #print(state)
            action = action + np.random.normal(0, self.agent.exploration_noise, size=self.agent.max_action)
            action = action.clip(-1*self.agent.max_action, self.agent.max_action)
            
            self.thrust_magnitude_Xp = action[0]*self.max_impulse
            self.thrust_magnitude_Yp = action[1]*self.max_impulse
            self.thrust_magnitude_Zp = action[2]*self.max_impulse

            #print(action)
            if current_time != 0.0:
                reward = self.agent.reward_computer.get_reward(state, self.last_action)
                self.agent.replay_buffer.add((self.last_state, action, reward, state, float(False)))
                self.agent.episode_reward += reward
                self.counter += 1

            self.last_state = state
            self.last_action = action
            self.processed_time = current_time      
