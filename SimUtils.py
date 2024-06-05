from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup

class SimSettings:
    def __init__(self):
        self.simulation_start_epoch = 0.0  # s
        self.global_frame_origin = 'Earth'
        self.global_frame_orientation = 'J2000'
        self.max_simtime = 10.0*60.0            # 10 minutes
        self.bodies_to_propagate = ['Chaser']
        self.central_bodies = ['Earth']
        self.integrator_stepsize = 0.1
        self.propagator = "Encke"
        self.bodies = self.get_environment_settings()
        self.acceleration_models = self.get_acceleration_settings()
        self.integrator_settings = self.get_integrator_settings()
        self.dep_vars_to_save = self.get_dependent_variables_to_save()
        self.termination_settings = self.get_termination_settings()

    def get_environment_settings(self):
        bodies_to_create = ['Earth']

        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            self.global_frame_origin,
            self.global_frame_orientation)

        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(self.global_frame_orientation, 'IAU_Earth', 'IAU_Earth', self.simulation_start_epoch)
        bodies = environment_setup.create_system_of_bodies(body_settings)

        self.add_chaser_body(bodies)
        self.add_target_body(bodies)

        return bodies
    
    def add_chaser_body(self, bodies):
        bodies.create_empty_body('Chaser')
        bodies.get_body('Chaser').set_constant_mass(10e3) 
    
    def add_target_body(self, bodies):
        bodies.create_empty_body('Target')
        bodies.get_body('Target').set_constant_mass(450e3) 
        #environment_setup.add_rotation_model( bodies, 'Capsule',
        #                                    environment_setup.rotation_model.aerodynamic_angle_based(
        #                                        'Earth', 'J2000', 'CapsuleFixed', angle_function ))


   
    def get_acceleration_settings(self):
        acceleration_settings_on_vehicle = {'Earth': [propagation_setup.acceleration.point_mass_gravity()]}

        # Create acceleration models.
        acceleration_settings = {'Chaser': acceleration_settings_on_vehicle}
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
        dependent_variables_to_save = [propagation_setup.dependent_variable.altitude('Chaser', 'Earth')]
        return dependent_variables_to_save
    
    def setup_simulation(self, initial_state):
        propagator_settings = propagation_setup.propagator.translational(self.central_bodies,
                                                                     self.acceleration_models,
                                                                     self.bodies_to_propagate,
                                                                     initial_state,
                                                                     self.simulation_start_epoch,
                                                                     self.integrator_settings,
                                                                     self.termination_settings,
                                                                     self.propagator,
                                                                     self.dep_vars_to_save)
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