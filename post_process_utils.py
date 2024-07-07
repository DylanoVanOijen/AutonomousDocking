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
    axis.set_xlabel("t-t$_{0} [s]$")
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
    axis.set_xlabel("t-t$_{0} [s]$")
    axis.set_ylabel("$\Delta r_{i}$ [m]")
    axis.grid()
    axis.legend()
    return axis


def plot_applied_thrust(axis, dep_vars):
    time = dep_vars[:,0]
    n_times = len(time)
    target_states = inertial_states[:,1:7]

    tnw_to_inertial_array = dep_vars[:, 1:10]
    tnw_to_inertial_matrix = tnw_to_inertial_array.reshape(n_times,3,3)
    delta_vel_tnw = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), delta_state_inertial[i,3:6]) for i in range(n_times)])

    axis.plot(time, delta_vel_tnw[:,0], label = "T", color = "red", linestyle = "solid")
    axis.plot(time, delta_vel_tnw[:,1], label = "N", color = "green", linestyle = "solid")
    axis.plot(time, delta_vel_tnw[:,2], label = "W", color = "blue", linestyle = "solid")
    axis.set_xlabel("t-t$_{0} [s]$")
    axis.set_ylabel("$\Delta r_{i}$ [m]")
    axis.grid()
    axis.legend()
    return axis