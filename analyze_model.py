from post_process_utils import *
from TD3_agent import *
from sim_utils import *

from tudatpy.kernel import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array

folder = "./tmp/td3/"

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


initial_cartesian_state = sim_settings.get_randomized_chaser_state()
prop = sim_settings.setup_simulation(initial_cartesian_state)

agent.load_models()

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    sim_settings.bodies, prop)

states = dynamics_simulator.state_history
dep_vars = dynamics_simulator.dependent_variable_history

states_array = result2array(states)
dep_vars_array = result2array(dep_vars)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
ax1 = plot_trajectory_2d(ax1, states_array, dep_vars_array)
ax2 = plot_velocity_2d(ax2, states_array, dep_vars_array)
ax3 = plot_thrust_body_frame(ax3, dep_vars_array)
ax4 = plot_thrust_TNW_frame(ax4, dep_vars_array)
fig.tight_layout()
fig.savefig("./plots/model_analysis.png")