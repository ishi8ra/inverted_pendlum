# 鈴木君のMatlabコードに、Bbの項を付け加える

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# パラメータ
r = 0.6
g = 9.81
b = 1 # TODO 書き換え
phi = -0.15
theta = 0.1 # θ
gamma = 0.5 # γ TODO 書き換え

# System matrices
As = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, g/r, 0, -g/r, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, g/r, 0, -g/r, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
Bl = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]
])
Bb = np.array([
    [0, 0, 0], # ↓ Bls1
    [0, 0, b],
    [0, 0, 0],
    [0, 0, (theta-b)/r],
    [0, 0, 0],
    [0, 0, 0],  # ↓ Bls2
    [0, 0, -gamma],
    [0, 0, 0],
    [0, 0, (phi-gamma)/r],
    [0, 0, 0],
    [0, 0, 0],   # ↓ Bls3
    [0, 0, 0]
])


# 重みの設定
Q = np.diag([1]*12)
R = np.diag([1, 1, 1])

# LQR法?
P = solve_continuous_are(As, Bl, Q, R)
K = np.linalg.inv(R) @ Bl.T @ P

# シミュレーションの設定
Tf = 10
Ts = 0.05
T = np.arange(0, Tf, Ts)
N = len(T)

# Reference and initial state
q_ref = np.zeros((12, N))
q0 = np.array([0, 0, 0.1, 0, 0, 0, 0, -0.15, 0, 0, 0, 0])

# 制御入力とかを初期化
u = np.zeros((3, N))
q = np.zeros((12, N))
dq = np.zeros((12, N))
q[:, 0] = q0

# シミュレーションを回す ← なにやってるか分からんけど。
for k in range(N - 1):
    u[:, k] = -K @ (q[:, k] - q_ref[:, k])

    # Compute dq based on the system dynamics (translated from the MATLAB code)
    dq[:, k] = np.array([
        q[1, k],
        (g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) +
         u[2, k]) * np.sin(q[4, k]),
        q[3, k],
        -q[8, k] ** 2 * np.sin(q[2, k]) * np.cos(q[2, k]) + (1 / r) * (
            -((g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) +
              u[2, k]) * np.sin(q[4, k])) * np.cos(q[2, k])
            - (-(g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) + u[2, k]) * np.sin(
                q[9, k]) * np.cos(q[4, k])) * np.sin(q[7, k]) * np.sin(q[2, k])
            + (((g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) + u[2, k]) * np.cos(
                q[4, k]) * np.cos(q[9, k])) - g + g) * np.cos(q[7, k]) * np.sin(q[2, k])
        ),
        u[0, k],
        q[6, k],
        -(g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) +
          u[2, k]) * np.sin(q[9, k]) * np.cos(q[4, k]),
        q[8, k],
        2 * q[3, k] * q[8, k] * np.tan(q[2, k]) + (1 / r) * (1 / np.cos(q[2, k])) * (
            (-(g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) + u[2, k])
             * np.sin(q[9, k]) * np.cos(q[4, k])) * np.cos(q[7, k])
            + (((g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) + u[2, k]) *
               np.cos(q[4, k]) * np.cos(q[9, k])) - g + g) * np.sin(q[7, k])
        ),
        u[1, k] / np.cos(q[4, k]),
        q[11, k],
        (g + 0.5 * g * (q[4, k] ** 2 + q[9, k] ** 2) +
         u[2, k]) * np.cos(q[4, k]) * np.cos(q[9, k]) - g
    ])

    q[:, k + 1] = q[:, k] + dq[:, k] * Ts 

# グラフを描画
plt.figure()
plt.plot(T, q[0, :], label='x')
plt.plot(T, q[6, :], label='y')
plt.plot(T, q[11, :], label='z')
plt.plot(T, q[2, :], label='θ')
plt.plot(T, q[7, :], label='Φ')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.grid(True)
plt.show()
