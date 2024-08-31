Hello, welcome to the github page for my final project for the TUDelft AE4350 "Bio-inspired Intelligence and learning for Aerospace Applications" course. For this assignment I implemented a TD3 algorithm for a 3DOF translational spacecraft docking problem. The file structure is as follows:

*/TrainingOutputs/* contains, as the name suggests, training outputs. In addition, the plots from the sensitivity analysis can be found here as well, under /TrainingOutputs/sensitivity_analysis/results.

*/plots/* contains the rest of the plots used in the report.

*/storage/* contains folders and files used during the making of this project, but not used anymore.

*TD3_agent.py* contains the class that computes the reward for a given state, and contains the agent class and all the code related to updating the networks using TD3.

*TD3_models.py* contains the replay buffer class, as well as the pytorch models for the actor and critic networks.

*analyze_parameter_combinations.py* is an analysis script to produce the results of the sensitivity analysis.

*analyze_reward_parameter_combinations.py* is analysis script to analyze the results of varying multiple parameters at one to see the best combination (not used in report).

*analyze_training.py* is an analysis script that analyzes the main training performance and the best trained model.

*hyperpar_cominations(_2, _3).py* are scripts to run different parameter combinations together and outputs the best models after training each parameter combination (not directly used in report).

*monte_carlo_analysis.py* performs a Monte-Carlo analysis on a specified model and analyzes the results right away as well.

*post_process_utils.py* contains functions to make all kind of plots used in the analysis.

*reward_tester.py* sanity check script to verify correct functioning of the reward strategy (not directly used in report).

*sensitivity_analysis.py* script to change parameter values and perform new training. Only creates training outputs, and needs analyze_parameter_combinations to perform analysis of results.

*sim_utils.py* contains all the code simulating the dynamics of the chaser and target vehicles, as well as being an interface between the tudat simulation and the pytorch models.


Note: when trying to run a pretrained model, it could happen that the code complains about not being able to load a model when a GPU is (or is not) unexpectely available. This can be fixed by going to the *TD3_models.py* file and swapping the commented line on 73 for 74, and similarly on 107 and 108 to force load to CPU.

*test_main.py* performs training for a single set of parameters. The test in the name is a bit of a misnomer, as for analyzing a single set of settings it is the main script to train a model.

*training_utils.py* contains a class to setup the training process for a given set of parameters, to make life a lot easier...
