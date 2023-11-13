import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# Parameters
g = 9.81

# Simplified system matrices for drone (assuming a basic model without pitch and roll)
A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, -g, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Weight matrices
Q = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([1, 1, 1])

# Solve Riccati equation
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Simulation setup
Tf = 10  # Simulation time 10 seconds
Ts = 0.05  # Sampling time 0.05 seconds
T = np.arange(0, Tf, Ts)
N = len(T)  # Total simulation steps

# Reference and initial state
q_ref = np.zeros((6, N))
q0 = np.array([0, 0, 0.1, 0, 0, 0])

# Initialize control inputs and states
u = np.zeros((3, N))
q = np.zeros((6, N))
dq = np.zeros((6, N))
q[:, 0] = q0

# Simulation loop
for k in range(N - 1):
    u[:, k] = -K @ (q[:, k] - q_ref[:, k])

    # Update dq based on the simplified system dynamics
    dq[:, k] = np.array([
        q[3, k],
        q[4, k],
        q[5, k],
        u[0, k],
        u[1, k],
        u[2, k] - g
    ])

    q[:, k + 1] = q[:, k] + dq[:, k] * Ts

# Plot the graph
plt.figure()
plt.plot(T, q[0, :], label='x')
plt.plot(T, q[1, :], label='y')
plt.plot(T, q[2, :], label='z')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.grid(True)
plt.show()
