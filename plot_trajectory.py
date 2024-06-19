import matplotlib.pyplot as plt
import numpy as np

vehicle_states = np.loadtxt("./PropagationHistory.dat")
time = vehicle_states[:,0]
n_times = len(time)
chaser_states = vehicle_states[:,1:7]
target_states = vehicle_states[:,7:13]
delta_state_inertial = chaser_states-target_states

dep_vars = np.loadtxt("./PropagationHistory_DependentVariables.dat")
tnw_to_inertial_array = dep_vars[:, 1:10]
tnw_to_inertial_matrix = tnw_to_inertial_array.reshape(n_times,3,3)
delta_pos_tnw = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), delta_state_inertial[i,0:3]) for i in range(n_times)])
delta_vel_tnw = np.array([np.matmul(np.transpose(tnw_to_inertial_matrix[i, :, :]), delta_state_inertial[i,3:6]) for i in range(n_times)])


""" plt.plot(time, chaser_states[:,0], label = "Chaser X", color = "red", linestyle = "solid")
plt.plot(time, chaser_states[:,1], label = "Chaser Y", color = "green", linestyle = "solid")
plt.plot(time, chaser_states[:,2], label = "Chaser Z", color = "blue", linestyle = "solid")
plt.plot(time, target_states[:,0], label = "Target X", color = "red", linestyle = "dashed")
plt.plot(time, target_states[:,1], label = "Target Y", color = "green", linestyle = "dashed")
plt.plot(time, target_states[:,2], label = "Target Z", color = "blue", linestyle = "dashed")
plt.xlabel("t-t$_{0} [s]$")
plt.ylabel("r$_{i}$ [m]")
plt.legend()
plt.show()


plt.plot(time, delta_state_inertial[:,0], label = "X", color = "red", linestyle = "solid")
plt.plot(time, delta_state_inertial[:,1], label = "Y", color = "green", linestyle = "solid")
plt.plot(time, delta_state_inertial[:,2], label = "Z", color = "blue", linestyle = "solid")
plt.xlabel("t-t$_{0} [s]$")
plt.ylabel("$\Delta r_{i}$ [m]")
plt.legend()
plt.show() """

plt.plot(time, delta_pos_tnw[:,0], label = "T", color = "red", linestyle = "solid")
plt.plot(time, delta_pos_tnw[:,1], label = "N", color = "green", linestyle = "solid")
plt.plot(time, delta_pos_tnw[:,2], label = "W", color = "blue", linestyle = "solid")
plt.xlabel("t-t$_{0} [s]$")
plt.ylabel("$\Delta r_{i}$ [m]")
plt.legend()
plt.show()
