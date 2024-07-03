# Module imports
from sim_utils import *
from agent import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt

# General imports
import numpy as np
import os
import time

if __name__ == '__main__':    
    n_games = 10    # number of simulations to run
    N = 20          # training interval / size of replay buffer, must be smaller than epoch duration   
    batch_size = 5  # size of the minibacthes
    n_epochs = 4    # How many times to iterate over minbatches during traing epoch
    alpha = 0.0003

    action_space_size = 6   # thrust in each direction
    obsservation_space_size = 6 # pos and vel in TNW frame of Target

    agent = Agent(n_actions=action_space_size, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=obsservation_space_size)
    
    altitude = 450E3 # meter
    target_kepler_orbit = np.array([6378E3+altitude, 0, 0, 0, 0, 0])
    sim_settings = SimSettings(target_kepler_orbit)
    sim_settings.chaser_GNC.add_agent(agent)
 
    score_history = []

    learning_steps = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        initial_cart_state = sim_settings.get_randomized_chaser_state()
        prop = sim_settings.setup_simulation(initial_cart_state)
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learning_steps += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)


