# Module imports
from sim_utils import *
from TD3_agent import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array

# General imports
import numpy as np
import os
import time

if __name__ == '__main__':    
    ###### Hyperparameters ######
    moving_avg_length = 10  # number of episodes over which to compute moving average
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr_actor = 0.0001            # learning rate of actor = alpha
    lr_critic = 0.0001           # learning rate of critic = beta
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.1          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # number of simulations to run
    n_iters = 100               # Number of training iterations per episode

    fc1_dim = 400               # Number of nodes in fully connected linear layer 1
    fc2_dim = 300               # Number of nodes in fully connected linear layer 2

    approach_direction = "pos_R-bar"    # choose from pos/neg and R, V and Z-bar (dynamics of Z-bar least intersting)
    reward_type = "simple"           # choose from simple, full or ...

    action_space_size = 3   # for each direction, pos, neg or no thrust
    observation_space_size = 6 # pos and vel in TNW frame of Target

    # Setting seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)   


    # Creating agent
    agent = Agent(alpha=lr_actor, beta=lr_critic, state_dim=observation_space_size, action_dim=action_space_size, 
                  fc1_dim=fc1_dim, fc2_dim=fc2_dim, max_action=1, batch_size=batch_size, gamma=gamma, 
                  polyak=polyak, policy_noise=policy_noise, noise_clip=noise_clip, policy_delay=policy_delay,
                  exploration_noise=exploration_noise, approach_direction=approach_direction, reward_type=reward_type)
    
    # Sim settings
    altitude = 450E3 # meter
    target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
    sim_settings = SimSettings(target_kepler_orbit)
    sim_settings.chaser_GNC.add_agent(agent)
 
    moving_reward_hist = []
    total_reward_hist = []

    best_reward = 0

    # run simulations
    for episode in range(max_episodes):
        initial_cartesian_state = sim_settings.get_randomized_chaser_state()
        prop = sim_settings.setup_simulation(initial_cartesian_state)
        score = 0

        # Create simulation object and propagate dynamics.
        t1 = time.process_time()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            sim_settings.bodies, prop)
        
        states = dynamics_simulator.state_history
        dep_vars = dynamics_simulator.dependent_variable_history

        states_array = result2array(states)
        dep_vars_array = result2array(dep_vars)

        t2 = time.process_time()
        agent.update(n_iters)
        t3 = time.process_time()


        total_reward_hist.append(agent.episode_reward)
        moving_reward_hist.append(agent.episode_reward)
        if episode+1 > moving_avg_length:
            moving_reward_hist.pop(0)

        if agent.episode_reward > best_reward:
            agent.save_models()
            best_reward = agent.episode_reward

        #agent.save_models()
        print(f"Episode: {episode}, Reward = {agent.episode_reward:.1f}, Mean {moving_avg_length} reward = {np.mean(moving_reward_hist):.1f}, proptime = {t2-t1:.1f}, traintime = {t3-t2:.1f}")

        agent.episode_reward = 0
        

