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
    
class Trainer:
    def __init__(self, settings:dict, save_folder):
        self.fig, ((self.ax1, self.ax2),(self.ax3, self.ax4))= plt.subplots(2,2, figsize=(12,8))
        #self.fig2, ((self.ax2),(self.ax3))= plt.subplots(2,1, figsize=(8,8))

        self.settings = settings
        
        # Setting seed
        torch.manual_seed(settings["random_seed"])
        np.random.seed(settings["random_seed"])  

        # Creating agent
        self.agent = Agent( alpha=settings["lr_actor"], beta=settings["lr_critic"], 
                        state_dim=settings["observation_space_size"], action_dim=settings["action_space_size"], 
                        fc1_dim=settings["fc1_dim"], fc2_dim=settings["fc2_dim"], 
                        max_action=settings["max_action"], buffer_size=settings["buffer_size"], batch_size=settings["batch_size"], 
                        gamma=settings["gamma"], tau=settings["tau"], 
                        policy_noise=settings["policy_noise"], noise_clip=settings["noise_clip"], 
                        policy_delay=settings["policy_delay"], exploration_noise=settings["exploration_noise"], 
                        approach_direction=settings["approach_direction"], 
                        reward_type=settings["reward_type"], reward_parameters=settings["reward_parameters"], 
                        docking_ports=settings["docking_port_locations"], docking_settings=settings["docking_settings"],warmup=settings["warmup"],
                        save_folder=save_folder)
    
        # Sim settings
        self.altitude = self.settings["sim_settings"]["orbit_height"]
        self.target_kepler_orbit = np.array([6378E3+self.altitude, 0, 0, 0, 0, 0])
        self.sim_settings = SimSettings(self.target_kepler_orbit, self.agent, settings["reward_type"])
    
        self.moving_reward_hist = np.array([])
        self.total_reward_hist = []

        self.best_reward = 0.0 
        self.past_mean_return = 0
        self.difficulty = 0
        self.n_iters = settings["n_iters"]
        self.save_each_episode = settings["save_each_episode"]
        self.save_folder = save_folder
        self.show_plots = settings["show_plots"]

    def start_training(self):
        for episode in range(self.settings["max_episodes"]):
            self.run_episode(episode)

        self.fig.savefig(self.save_folder+"reward_history.png")
        plt.close(self.fig)

    def run_episode(self, episode):
        initial_cartesian_state = self.sim_settings.get_randomized_chaser_state(self.difficulty)
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
        #self.agent.update(self.n_iters)            # always fixed number of training iterations per simulation
        #print(len(dep_vars.keys())-1)
        #self.agent.update(len(dep_vars.keys())-1)   # --> alternative approach
        #t3 = time.process_time()

        self.total_reward_hist.append(self.agent.episode_reward)
        self.moving_reward_hist = np.append(self.moving_reward_hist, self.agent.episode_reward)

        if len(self.moving_reward_hist) > 10:
            self.moving_reward_hist = np.delete(self.moving_reward_hist, 0)

        mean = np.mean(self.moving_reward_hist)
        if episode > 10:
            if self.difficulty == 0 and mean > 220:
                print("Now moving to difficulty 1")
                self.difficulty = 1
                self.moving_reward_hist = np.array([])
                #self.agent.decrease_lr(5)

            elif self.difficulty == 1 and mean > 250:
                print("Now moving to difficulty 2")
                self.difficulty = 2
                #self.agent.decrease_lr(2)

            elif self.difficulty == 1 and mean > 250:
                print("Now moving to difficulty 3")
                self.difficulty = 3

        if episode == 0:
            self.best_reward = self.agent.episode_reward

        if self.save_each_episode:
            self.agent.save_models()
        elif not self.save_each_episode and self.agent.episode_reward >= self.best_reward:
            self.agent.save_models()
            self.best_reward = self.agent.episode_reward

        print(f"Episode: {episode}, Reward = {self.agent.episode_reward:.1f}, proptime = {t2-t1:.1f}")

        self.agent.episode_reward = 0
        
        self.ax1 = plot_training_performance(self.ax1, self.total_reward_hist)

        plt.tight_layout()
        if self.show_plots:
            self.ax2.cla()
            self.ax3.cla()
            self.ax4.cla()
            self.ax2 = plot_trajectory_2d(self.ax2, states_array, dep_vars_array)
            self.ax4 = plot_velocity_2d(self.ax4, states_array, dep_vars_array)
            self.ax3 = plot_thrust_body_frame(self.ax3, dep_vars_array)
            plt.draw()
            plt.pause(0.5)
