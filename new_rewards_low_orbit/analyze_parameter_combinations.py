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

pars = {"random_seed": { "name" : "Random seed",
                        "label" : "seed =",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
    
        "n_iters" : {   "name" : "Number of iterations",
                        "label" : "n",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
        "batch_size": { "name" : "Batch size",
                        "label" : "m",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
        "gamma": {      "name" : "Discount factor",
                        "label" : r"$\gamma$",
                        "n_values" : 3,
                        "linestyle" : "solid"
                    },
        "lr": {      "name" : "Learning rate",
                        "label" : r"$\alpha$",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
        "exploration_noise": { "name" : "Exploration noise",
                        "label" : r"$\sigma_{e}$",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
        "policy_noise": { "name" : "Policy noise",
                        "label" : r"$\sigma_{p}$",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
        "polyak": { "name" : "Polyak",
                        "label" : r"$\rho$",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    },
        "policy_delay": { "name" : "Policy delay",
                        "label" : "d",
                        "n_values" : 2,
                        "linestyle" : "solid"
                    }
        }


main_dir = pre_dir + "main/"
with open(main_dir+'main_settings.txt', 'r') as convert_file:
    main_settings = json.load(convert_file)

main_history = np.loadtxt(main_dir+"reward_history_data.txt")
main_mean_history = compute_mean(main_history)

fig_all, axes =  plt.subplots(3,3, figsize=(12,10))
axes = axes.flatten()

counter = 0
for parameter in pars.keys():
    par_id = parameter
    par_info = pars[par_id]

    title_name = par_info["name"]

    if par_id == "lr":
        main_label = par_info["label"] + " = " + str(main_settings["lr_actor"])
    else:
        main_label = par_info["label"] + " = " + str(main_settings[par_id])

    fig, ax =  plt.subplots(1,1, figsize=(7,5))
    ax = plot_multi_training_performance(ax, main_history, main_label, par_info["linestyle"], title_name, main_mean_history)
    axes[counter] = plot_multi_training_performance(axes[counter], main_history, main_label, par_info["linestyle"], title_name, main_mean_history)


    for i in range(1, par_info["n_values"]+1):
        par_value_dir = pre_dir + par_id + f"_{i}/"

        with open(par_value_dir+'settings.txt', 'r') as convert_file:
            settings = json.load(convert_file)

        par_value_history = np.loadtxt(par_value_dir+"reward_history_data.txt")
        par_mean_history = compute_mean(par_value_history)

        if par_id == "lr":
            label = par_info["label"] + " = " + str(settings["lr_actor"])
        else:
            label = par_info["label"] + " = " + str(settings[par_id])
        ax = plot_multi_training_performance(ax, par_mean_history, label, par_info["linestyle"], title_name, par_mean_history)
        axes[counter] = plot_multi_training_performance(axes[counter], par_mean_history, label, par_info["linestyle"], title_name, par_mean_history)


    fig.tight_layout()
    fig.savefig(analysis_dir+par_id+".png")
    ax.cla()
    fig.clf()

    counter += 1

for i in range(9):
    if i not in [0,3,6]:
        axes[i].set_ylabel("")
    if i not in [6,7,8]:
        axes[i].set_xlabel("")


fig_all.suptitle("5-episode moving average return for different parameter values", fontsize=16)
#fig_all.subplots_adjust(top=0.9)
fig_all.tight_layout()
fig_all.savefig(analysis_dir+"all.png")
