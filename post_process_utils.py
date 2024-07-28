import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory_2d(axis, inertial_states, dep_vars):
    time = inertial_states[:,0]
    n_times = len(time)
    target_states = inertial_states[:,1:7]
    chaser_states = inertial_states[:,7:13]
    delta_state_inertial = chaser_states-target_states

    tnw_to_inertial_array = dep_vars[:, 1:10]
    tnw_to_inertial_matrix = tnw_to_inertial_array.reshape(n_times,3,3)
    delta_pos_tnw = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), delta_state_inertial[i,0:3]) for i in range(n_times)])

    axis.plot(time, delta_pos_tnw[:,0], label = "T", color = "red", linestyle = "solid")
    axis.plot(time, delta_pos_tnw[:,1], label = "N", color = "green", linestyle = "solid")
    axis.plot(time, delta_pos_tnw[:,2], label = "W", color = "blue", linestyle = "solid")
    axis.set_xlim(time[0], time[-1])
    axis.plot([time[0], time[-1]], [0,0], label = "Port loc T", color = "red", linestyle = "dotted")
    axis.plot([time[0], time[-1]], [0,0], label = "Port loc N", color = "green", linestyle = "dotted")
    axis.plot([time[0], time[-1]], [0,0], label = "Port loc W", color = "blue", linestyle = "dotted")
    axis.set_title("Position of chaser in target TNW frame")
    axis.set_xlabel("t-t$_{0}$ [s]")
    axis.set_ylabel("$\Delta r_{i}$ [m]")
    axis.grid()
    axis.legend()

    return axis

def plot_velocity_2d(axis, inertial_states, dep_vars):
    time = inertial_states[:,0]
    n_times = len(time)
    target_states = inertial_states[:,1:7]
    chaser_states = inertial_states[:,7:13]
    delta_state_inertial = chaser_states-target_states

    tnw_to_inertial_array = dep_vars[:, 1:10]
    tnw_to_inertial_matrix = tnw_to_inertial_array.reshape(n_times,3,3)
    delta_vel_tnw = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), delta_state_inertial[i,3:6]) for i in range(n_times)])

    axis.plot(time, delta_vel_tnw[:,0], label = "T", color = "red", linestyle = "solid")
    axis.plot(time, delta_vel_tnw[:,1], label = "N", color = "green", linestyle = "solid")
    axis.plot(time, delta_vel_tnw[:,2], label = "W", color = "blue", linestyle = "solid")
    axis.set_xlim(time[0], time[-1])
    axis.set_title("Velocity of chaser in target TNW frame")
    axis.set_xlabel("t-t$_{0}$ [s]")
    axis.set_ylabel("$\Delta v_{i}$ [m]")
    axis.grid()
    axis.legend()

    return axis

def plot_thrust_body_frame(axis, dep_vars):
    time = dep_vars[:,0]
    n_times = len(time)

    inertial_thrust = dep_vars[:,19:22]
    inertial_to_body_array = dep_vars[:, 10:19]
    inertial_to_body_matrix = inertial_to_body_array.reshape(n_times,3,3)
    body_thrust = np.array([np.matmul(inertial_to_body_matrix[i, :, :], inertial_thrust[i]) for i in range(n_times)])

    axis.plot(time, body_thrust[:,0], label = "X", color = "red", linestyle = "solid")
    axis.plot(time, body_thrust[:,1], label = "Y", color = "green", linestyle = "solid")
    axis.plot(time, body_thrust[:,2], label = "Z", color = "blue", linestyle = "solid")
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(-0.05,0.05)
    axis.set_title("Thrust in chaser body frame")
    axis.set_xlabel("t-t$_{0}$ [s]")
    axis.set_ylabel("a$_{i}$ [m]")
    axis.grid()
    axis.legend()

    return axis

def plot_thrust_TNW_frame(axis, dep_vars):
    time = dep_vars[:,0]
    n_times = len(time)

    inertial_thrust = dep_vars[:,19:22]
    tnw_to_inertial_array = dep_vars[:, 1:10]
    tnw_to_inertial_matrix = tnw_to_inertial_array.reshape(n_times,3,3)
    thrust_TNW = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), inertial_thrust[i]) for i in range(n_times)])

    axis.plot(time, thrust_TNW[:,0], label = "T", color = "red", linestyle = "solid")
    axis.plot(time, thrust_TNW[:,1], label = "N", color = "green", linestyle = "solid")
    axis.plot(time, thrust_TNW[:,2], label = "W", color = "blue", linestyle = "solid")
    axis.set_xlim(time[0], time[-1])
    axis.set_title("Thrust in target TNW frame")
    axis.set_xlabel("t-t$_{0}$ [s]")
    axis.set_ylabel("a$_{i}$ [m]")
    axis.grid()
    axis.legend()
    return axis


def plot_trajectory_3d(axis, inertial_states, dep_vars, port_loc):
    target_states = inertial_states[:,1:7]
    chaser_states = inertial_states[:,7:13]
    delta_state_inertial = chaser_states-target_states
    n_times = target_states.shape[0]

    tnw_to_inertial_array = dep_vars[:, 1:10]
    tnw_to_inertial_matrix = tnw_to_inertial_array.reshape(n_times, 3,3)
    delta_pos_tnw = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), delta_state_inertial[i,0:3]) for i in range(n_times)])

    max_value = np.max(np.abs(delta_pos_tnw))

    axis.plot3D(delta_pos_tnw[:,0], delta_pos_tnw[:,2], delta_pos_tnw[:,1], color='blue', label="Trajectory")
    axis.scatter(port_loc[0], port_loc[2], port_loc[1], color='black', label="Target Docking Port")
    axis.scatter(delta_pos_tnw[0,0], delta_pos_tnw[0,2], delta_pos_tnw[0,1], color='green', label="Initial position")
    axis.scatter(delta_pos_tnw[-1,0], delta_pos_tnw[-1,2],delta_pos_tnw[-1,1], color='red', label="Final position")
    axis.set_title("Position of chaser in target TNW frame")
    axis.set_xlabel("T [x]")
    axis.set_ylabel("W [m]")
    axis.set_zlabel("N [m]")
    axis.set_xlim(-max_value,max_value)
    axis.set_ylim(-max_value,max_value)
    axis.set_zlim(-max_value,max_value)
    axis.legend()
    axis.grid()

    return axis


def plot_training_performance(ax, reward):
    episodes = np.arange(0,len(reward))

    #print(episodes, reward)
    #print(episodes, moving_avg_reward)

    ax.clear()
    ax.plot(episodes, reward, label = "Episode reward")
    ax.set_xlabel("Episode [-]")
    ax.set_ylabel("Reward value [-]")
    ax.set_xlim(0,len(episodes))
    #y_low = ax.get_ylim()[0]
    #ax.set_ylim((45,60))
    #ax.set_yscale('log')
    ax.legend()
    ax.grid()

    return ax
