import matplotlib.pyplot as plt
import numpy as np

vehicle_states = np.loadtxt("./PropagationHistory.dat")
time = vehicle_states[:,0]
chaser_states = vehicle_states[:,1:7]
target_states = vehicle_states[:,7:13]
delta_state = chaser_states-target_states

print(vehicle_states[0,:])
print(chaser_states[0])
print(target_states[0])

plt.plot(time, chaser_states[:,0], label = "Chaser X", color = "red", linestyle = "solid")
plt.plot(time, chaser_states[:,1], label = "Chaser Y", color = "green", linestyle = "solid")
plt.plot(time, chaser_states[:,2], label = "Chaser Z", color = "blue", linestyle = "solid")
plt.plot(time, target_states[:,0], label = "Target X", color = "red", linestyle = "dashed")
plt.plot(time, target_states[:,1], label = "Target Y", color = "green", linestyle = "dashed")
plt.plot(time, target_states[:,2], label = "Target Z", color = "blue", linestyle = "dashed")
plt.xlabel("t-t$_{0} [s]$")
plt.ylabel("r$_{i}$ [m]")
plt.legend()
plt.show()


plt.plot(time, delta_state[:,0], label = "X", color = "red", linestyle = "solid")
plt.plot(time, delta_state[:,1], label = "Y", color = "green", linestyle = "solid")
plt.plot(time, delta_state[:,2], label = "Z", color = "blue", linestyle = "solid")
plt.xlabel("t-t$_{0} [s]$")
plt.ylabel("$\Delta r_{i}$ [m]")
plt.legend()
plt.show()
