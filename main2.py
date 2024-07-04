# Module imports
from sim_utils import *
from TD3_agent import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt

# General imports
import numpy as np
import os
import time

if __name__ == '__main__':    


    ###### Hyperparameters ######
    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr_actor = 0.001            # learning rate of actor = alpha
    lr_critic = 0.001           # learning rate of critic = beta
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # number of simulations to run
    max_timesteps = 2000        # max timesteps in one episode

    fc1_dim = 400               # Number of nodes in fully connected linear layer 1
    fc2_dim = 400               # Number of nodes in fully connected linear layer 2

    action_space_size = 9   # for each direction, pos, neg or no thrust
    observation_space_size = 6 # pos and vel in TNW frame of Target

    # Setting seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)   

    # Creating agent
    agent = Agent(alpha=lr_actor, beta=lr_critic, state_dim=observation_space_size, action_dim=action_space_size, 
                  fc1_dim=fc1_dim, fc2_dim=fc2_dim, max_action=1, batch_size=batch_size, gamma=gamma, 
                  polyak=polyak, policy_noise=policy_noise, noise_clip=noise_clip, policy_delay=policy_delay,
                  exploration_noise=exploration_noise)
    
    # Sim settings
    altitude = 450E3 # meter
    target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
    sim_settings = SimSettings(target_kepler_orbit)
    sim_settings.chaser_GNC.add_agent(agent)
 
    score_history = []

    learning_steps = 0
    avg_reward = 0
    ep_reward = 0
    best_reward = 0

    # run simulations
    for i in range(max_episodes):
        initial_cartesian_state = sim_settings.get_randomized_chaser_state()
        prop = sim_settings.setup_simulation(initial_cartesian_state)
        score = 0

        # Create simulation object and propagate dynamics.
        t1 = time.process_time()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            sim_settings.bodies, prop)
        t2 = time.process_time()
        print("Time = ", t2-t1)

        # update using the len(t)-1, so all steps
        agent.update(len())

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        #print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
        #        'time_steps', n_steps, 'learning_steps', learn_iters)


