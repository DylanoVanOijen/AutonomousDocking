# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "hyperparameter_combinations/"
pre_dir = main_dir+sub_dir

if (not os.path.isdir(pre_dir)):
    os.mkdir(pre_dir)

docking_port_locations = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([-2, -2 , 0])
        }

sim_settings = {
    "step_size": 1.0,
    "orbit_height" : 100000e3   # m
}

docking_settings = {
    "KOS_size" : 50,
    "corridor_base_radius": 1,   # meter
    "corridor_angle" : np.deg2rad(20), 
    "max_distance" : 50,
    "max_offdir_pos" : 0.2,
    "max_offdir_vel" : 0.01,
    "ideal_dir_vel" : 0.2,
    "min_dir_vel" : 0.1,
    "max_dir_vel" : 0.2  
}

reward_parameters = {
    "eta" : 10,
    "kappa" : 0.1,  # tanh scaling
    "lamda" : 0.1,  # 
    "mu" : 0.05,
    "corridor_penalty" : 10,
    "far_away_penalty" : 50,
    "docking_pos_bonus" : 10,
    "docking_vel_bonus" : 10,
    "docking_pos_bonus_scaling" : 10,
    "docking_vel_bonus_scaling" : 10,
}

settings = {"random_seed":42,
            "max_action":1,
            "gamma": 0.99,
            "buffer_size": 3e3,
            "batch_size": 50,              # num of transitions sampled from replay buffer
            "lr_actor":10**(-6),              # learning rate of actor = alpha
            "lr_critic":10**(-6),             # learning rate of critic = beta
            "exploration_noise":0.01, 
            "polyak":0.995,                 # target policy update parameter (1-tau)
            "policy_noise":0.01,             # target policy smoothing noise
            "noise_clip":0.2,
            "policy_delay":2,               # delayed policy updates parameter
            "max_episodes":250,             # number of simulations to run
            "n_iters":500,                   # Number of training iterations per episode (not used anymore)
            "fc1_dim":128,                  # Number of nodes in fully connected linear layer 1
            "fc2_dim":128,                  # Number of nodes in fully connected linear layer 2
            "save_each_episode":False,        # Flag to save the models after each epoch instead of only when the results improved
            "approach_direction":"pos_R-bar",# choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
            "reward_type":"full",          # choose from simple, full or ...
            "action_space_size":3,          # for each direction, pos, neg or no thrust
            "observation_space_size":6,     # pos and vel in TNW frame of Target
            "docking_port_locations":docking_port_locations,
            "docking_settings":docking_settings,
            "reward_parameters":reward_parameters,
            "sim_settings":sim_settings
            }
transitions = 60
pars_to_loop = {"buffer_size" : [20*transitions, 50*transitions, 150*transitions],
                "batch_size" : [2*transitions, 5*transitions, 10*transitions],
                "lr" : [10**(-7), 10**(-6), 10**(-5)],
                "n_iters": [10,100,1000]
                }

settings["exploration_noise"] = 0
settings["noise_clip"] = 0

seeds = [41, 42, 43]

option_counter = 0
n_options = len(pars_to_loop["buffer_size"])*len(pars_to_loop["batch_size"])*len(pars_to_loop["lr"])*len(pars_to_loop["n_iters"])
for buff_size in pars_to_loop["buffer_size"]:
    settings["buffer_size"] = buff_size

    for batch_size in pars_to_loop["batch_size"]:
        settings["batch_size"] = batch_size

        for lr in pars_to_loop["lr"]:
            settings["lr_actor"] = lr
            settings["lr_critic"] = lr

            for n_iters in pars_to_loop["n_iters"]:
                settings["n_iters"] = n_iters
                option_counter += 1

                save_dir = main_dir+sub_dir+f"option_{option_counter}/"
                seed_dir = save_dir + f"/seed_42/"

                print(f"Option: {option_counter}, buff size: {buff_size}, batch size: {batch_size}, lr: {lr}, n_iters {n_iters}")

                for seed in seeds:
                    settings["random_seed"] = seed

                    torch.manual_seed(settings["random_seed"])
                    np.random.seed(settings["random_seed"])  

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
                                            save_folder=seed_dir)
                        

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

                    final_reward = agent.episode_reward
                    print(f"Seed {seed}, episode return {final_reward:.2f}")