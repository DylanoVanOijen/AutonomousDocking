import pickle

# Module imports
from sim_utils import *
from TD3_agent import *
from post_process_utils import *

# Tudat imports
from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array

# General imports
import matplotlib.pyplot as plt
import numpy as np
import os
import time


class TrainSettings:
    def __init__(self, seed, batch_size, lr_actor, lr_critic, exploration_noise, polyak,
                 policy_noise, noise_clip, policy_delay, max_episodes, n_iters, fc1_dim, fc2_dim,
                 save_each_epoch, approach_direction, reward_type, action_space_size, observation_space_size):
    
        self.seed = seed
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.exploration_noise = exploration_noise
        self.polyak = polyak
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_episodes = max_episodes
        self.n_iters = n_iters
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.save_each_epoch = save_each_epoch
        self.approach_direction = approach_direction
        self.reward_type = reward_type
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as file:
            obj = cls.__new__(cls)
            obj.__dict__.update(pickle.load(file))
        return obj
    

class Trainer:
    def __init__(self, settings:dict):
        self.fig, ((self.ax1))= plt.subplots(1,1, figsize=(8,5))

        self.settings = settings
        
        # Setting seed
        torch.manual_seed(settings["random_seed"])
        np.random.seed(settings["random_seed"])  

        # Creating agent
        self.agent = Agent(  alpha=settings["lr_actor"], beta=settings["lr_critic"], 
                        state_dim=settings["observation_space_size"], action_dim=settings["action_space_size"], 
                        fc1_dim=settings["fc1_dim"], fc2_dim=settings["fc2_dim"], 
                        max_action=settings["max_action"], batch_size=settings["batch_size"], 
                        gamma=settings["gamma"], polyak=settings["polyak"], 
                        policy_noise=settings["policy_noise"], noise_clip=settings["noise_clip"], 
                        policy_delay=settings["policy_delay"], exploration_noise=settings["exploration_noise"], 
                        approach_direction=settings["approach_direction"], reward_type=settings["reward_type"])
    
        # Sim settings
        self.altitude = 450E3 # meter
        self.target_kepler_orbit = np.array([6378E3+self.altitude, 0, 0, 0, 0, 0])
        self.sim_settings = SimSettings(self.target_kepler_orbit)
        self.sim_settings.chaser_GNC.add_agent(self.agent)
    
        self.moving_reward_hist = []
        self.total_reward_hist = []

        self.best_reward = 0.0 
        self.episode = 0
        self.n_iters = settings["n_iters"]
        self.save_each_episode = settings["save_each_episode"]

    def start_training(self):
        for episode in range(self.settings["max_episodes"]):
            self.run_episode(episode)

    def run_episode(self, episode):
        initial_cartesian_state = self.sim_settings.get_randomized_chaser_state()
        prop = self.sim_settings.setup_simulation(initial_cartesian_state)

        # Create simulation object and propagate dynamics.
        t1 = time.process_time()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.sim_settings.bodies, prop)
        
        states = dynamics_simulator.state_history
        dep_vars = dynamics_simulator.dependent_variable_history

        states_array = result2array(states)
        dep_vars_array = result2array(dep_vars)

        t2 = time.process_time()
        self.agent.update(self.n_iters)
        t3 = time.process_time()

        self.total_reward_hist.append(self.agent.episode_reward)

        if episode == 0:
            self.best_reward = self.agent.episode_reward

        if self.save_each_episode:
            self.agent.save_models()
        elif not self.save_each_episode and self.agent.episode_reward >= self.best_reward:
            self.agent.save_models()
            self.best_reward = self.agent.episode_reward

        print(f"Episode: {episode}, Reward = {self.agent.episode_reward:.1f}, proptime = {t2-t1:.1f}, traintime = {t3-t2:.1f}")

        self.agent.episode_reward = 0
        
        self.ax1 = plot_training_performance(self.ax1, self.total_reward_hist)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)
