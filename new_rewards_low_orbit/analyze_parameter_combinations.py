# Module imports
from training_utils import *

# General imports
import json 

main_dir = "./TrainingOutputs/"
sub_dir = "sensitivity_analysis/"
pre_dir = main_dir+sub_dir
analysis_dir  = pre_dir+"results/"

if (not os.path.isdir(analysis_dir)):
    os.mkdir(analysis_dir)

dir_2d = analysis_dir+"2D/"
dir_3d = analysis_dir+"3D/"
dir_rewards = analysis_dir+"rewards/"
dir_MC = analysis_dir+"MC/"

""" if (not os.path.isdir(dir_2d)):
    os.mkdir(dir_2d)

if (not os.path.isdir(dir_3d)):
    os.mkdir(dir_3d)

if (not os.path.isdir(dir_rewards)):
    os.mkdir(dir_rewards)

if (not os.path.isdir(dir_MC)):
    os.mkdir(dir_MC) """

pars = {"n_iters" : {   "name" : "number of iteration",
                        "label" : "n",
                        "n_values" : 2
                    },
        "batch_size": { "name" : "batch size",
                        "label" : "m",
                        "n_values" : 2
                    },
        "gamma": {      "name" : "discount factor",
                        "label" : r"$\gamma$",
                        "n_values" : 3
                    },
        "lr": {      "name" : "learning rate",
                        "label" : r"$\alpha$",
                        "n_values" : 2
                    },
        "exploration_noise": { "name" : "exploration noise",
                        "label" : r"$\sigma_{e}$",
                        "n_values" : 2
                    },
        "policy_noise": { "name" : "policy noise",
                        "label" : r"$\sigma_{p}$",
                        "n_values" : 2
                    },
        "polyak": { "name" : "polyak",
                        "label" : r"$\rho$",
                        "n_values" : 2
                    }
        }

main_dir = pre_dir + "main/"

# Reading the data from the file and converting it back to a dictionary
with open(main_dir+'main_settings.txt', 'r') as convert_file:
    main_settings = json.load(convert_file)

main_history = np.loadtxt(main_dir+"reward_history_data.txt")

for parameter in pars.keys():
    par_id = parameter
    par_info = pars[par_id]

    if par_id == "lr":
        main_label = par_info["label"] + " = " + str(main_settings["lr_actor"])
    else:
        main_label = par_info["label"] + " = " + str(main_settings[par_id])

    fig, ax =  plt.subplots(1,1, figsize=(7,5))
    ax = plot_multi_training_performance(ax, main_history, main_label)

    for i in range(1, par_info["n_values"]+1):
        par_value_dir = pre_dir + par_id + f"_{i}/"

        with open(par_value_dir+'settings.txt', 'r') as convert_file:
            settings = json.load(convert_file)

        par_value_history = np.loadtxt(par_value_dir+"reward_history_data.txt")

        if par_id == "lr":
            label = par_info["label"] + " = " + str(settings["lr_actor"])
        else:
            label = par_info["label"] + " = " + str(settings[par_id])
        ax = plot_multi_training_performance(ax, par_value_history, label)

    fig.tight_layout()
    fig.savefig(analysis_dir+par_id+".png")





""" for  in range(1,n_options+1):
    folder = pre_dir+f"option_{i}/"

    # Reading the data from the file and converting it back to a dictionary
    with open(folder+'settings.txt', 'r') as convert_file:
        settings = json.load(convert_file)

    # Convert docking port locations back from list to NDarray
    port_locs = settings["docking_port_locations"]
    for port_name in port_locs:
        port_loc_array = np.array(port_locs[port_name])
        port_locs[port_name] = port_loc_array
    
    settings["docking_port_locations"] = port_locs
    settings["policy_noise"] = 0
    settings["exploration_noise"] = 0
    settings["noise_clip"] = 0

    #print(settings["conditinal_reward_w"])
    settings["reward_parameters"]["conditional_reward_w"] = settings["reward_parameters"]["conditinal_reward_w"]


    torch.manual_seed(settings["random_seed"])
    np.random.seed(settings["random_seed"])  

    #torch.manual_seed(42)
    #np.random.seed(42)  

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
    altitude = settings["sim_settings"]["orbit_height"]
    target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
    sim_settings = SimSettings(target_kepler_orbit, agent, settings["reward_type"])

    initial_cartesian_state = sim_settings.get_randomized_chaser_state(-1)
    prop = sim_settings.setup_simulation(initial_cartesian_state)

    agent.load_models()

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        sim_settings.bodies, prop)

    states = dynamics_simulator.state_history
    dep_vars = dynamics_simulator.dependent_variable_history

    states_array = result2array(states)
    dep_vars_array = result2array(dep_vars)

    final_reward = agent.episode_reward

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
    ax1 = plot_action(ax1, dep_vars_array)
    ax2 = plot_trajectory_2d(ax2, states_array, dep_vars_array)
    #ax1.plot(dep_vars_array[:,0], dep_vars_array[:,22], label = "x dep", color = "cyan", linestyle = "dashed")
    #ax1.plot(dep_vars_array[:,0], dep_vars_array[:,23], label = "y dep", color = "magenta", linestyle = "dashed")
    #ax1.plot(dep_vars_array[:,0], dep_vars_array[:,24], label = "z dep", color = "yellow", linestyle = "dashed")
    ax2.legend()
    ax4 = plot_velocity_2d(ax4, states_array, dep_vars_array)
    #ax2.plot(dep_vars_array[:,0], dep_vars_array[:,25], label = "x dep", color = "cyan", linestyle = "dashed")
    #ax2.plot(dep_vars_array[:,0], dep_vars_array[:,26], label = "y dep", color = "magenta", linestyle = "dashed")
    #ax2.plot(dep_vars_array[:,0], dep_vars_array[:,27], label = "z dep", color = "yellow", linestyle = "dashed")
    ax4.legend()
    #ax4 = plot_thrust_TNW_frame(ax4, dep_vars_array)
    fig.tight_layout()
    fig.savefig(dir_2d+f"option_{i}")


    port_loc = settings["docking_port_locations"]["pos_R-bar"]
    fig2 = plt.figure()
    ax_3d = fig2.add_subplot(projection='3d')
    ax_3d = plot_trajectory_3d(ax_3d, states_array, dep_vars_array, port_loc)
    fig2.tight_layout()
    fig2.savefig(dir_3d+f"option_{i}")


    reward_history = np.loadtxt(folder+"reward_history_data.txt")
    fig3, ax5 = plt.subplots(1,1, figsize=(8,6))
    ax5 = plot_training_performance(ax5, reward_history)
    fig3.tight_layout()
    fig3.savefig(dir_rewards+f"option_{i}") 


    # mc
    arrival_times = []
    off_axis_distances = []
    off_axis_velocities = []
    dir_velocities = []
    docking_successes = []

    n_samples = 100
    for j in range(n_samples):
        if j % (n_samples // 20) == 0:
            print(f"At {j / n_samples * 100}%")

        initial_cartesian_state = sim_settings.get_randomized_chaser_state(-1)
        prop = sim_settings.setup_simulation(initial_cartesian_state)
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        sim_settings.bodies, prop)

        states = dynamics_simulator.state_history
        dep_vars = dynamics_simulator.dependent_variable_history

        states_array = result2array(states)
        dep_vars_array = result2array(dep_vars)

        arrival_time, off_axis_distance, off_axis_velocity, dir_velocity, docking_success = compute_MC_statistics(states_array, dep_vars_array, settings)

        arrival_times.append(arrival_time)
        off_axis_distances.append(off_axis_distance)
        off_axis_velocities.append(off_axis_velocity)
        dir_velocities.append(dir_velocity)
        docking_successes.append(docking_success)


    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
    ax1 = plot_arrival_times(ax1, arrival_times, docking_successes, 50)
    ax2 = plot_off_axis_distances(ax2, off_axis_distances, docking_successes, 50)
    ax4 = plot_off_axis_velocities(ax4, off_axis_velocities, docking_successes, 50)
    ax3 = plot_dir_axis_velocities(ax3, dir_velocities, docking_successes, 50)
    fig3.tight_layout()
    fig3.savefig(dir_MC+f"option_{i}")

    sucess_rate = sum(docking_successes)/len(docking_successes)*100
    print(f"Option {i}, sucessrate = {sucess_rate:.2f}")

    #plt.show()
    plt.close() """