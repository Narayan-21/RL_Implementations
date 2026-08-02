"""
Compares three ways of solving the Bellman expectation equation
    
    V = R^pi + gamma * P^pi @ V

for a fixed policy pi, all of which are just different orderings of
applying the same gamma-contraction operator T^pi:

1. Jacobi-Style -> Synchronous, two array backups
2. Gauss-Seibel-style -> in-place, single-array backups (fixed sweep order)
3. Asynchronous DP -> in-place backups, RANDOM state order each sweap

All three provably converge to the unique fixed point V_pi (Banach fixed
point theorem, since ||T^pi V1 - T^pi V2||_inf <= gamma ||V1 - V2||_inf).
We verify this numerically and visualize convergence speed.

"""

import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINAL_STATES = {0, N_STATES - 1}
ACTIONS = ["up", "down", "left", "right"]
GAMMA = 0.9

def step(s, a):
    if s in TERMINAL_STATES:
        return s, 0.0 # Zero rewards in the terminal states
    r, c = divmod(s, GRID_SIZE)
    if a == "up":
        r = max(r-1, 0)
    elif a == "down":
        r = min(r+1, GRID_SIZE - 1)
    elif a == "left":
        c = max(c-1, 0)
    elif a == "right":
        c = min(c+1, GRID_SIZE - 1)
    s_next = r * GRID_SIZE + c
    reward = -1.0
    return s_next, reward

def build_Ppi_Rpi(policy):
    """
    Build the state-transition matrix P^pi (n x n) and expected reward
    vector R^pi (n,) for a given stochastic policy: policy[s][a] = prob.
    """
    P = np.zeros((N_STATES, N_STATES))
    R = np.zeros(N_STATES)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            P[s, s] = 1.0
            R[s] = 0.0
            continue
        for a in ACTIONS:
            prob_a = policy[s][a]
            s_next, r = step(s, a)
            P[s, s_next] += prob_a
            R[s] += prob_a * r
    return P, R


# Random policy -> 0.25 prob for each of the 4 actions, every state
policy = [{a: 0.25 for a in ACTIONS} for _ in range(N_STATES)]
P_pi, R_pi = build_Ppi_Rpi(policy)

A = np.eye(N_STATES) - GAMMA * P_pi
V_true = np.linalg.solve(A, R_pi)
V_true[list(TERMINAL_STATES)] = 0.0  # pin terminal values exactly

# Bellman Operator (applied to one state at a time)

def bellman_backup(s, V_read):
    """One-state Bellman expectation backup, reading successor values from V_read."""
    if s in TERMINAL_STATES:
        return 0.0
    total = 0.0
    for a in ACTIONS:
        prob_a = policy[s][a]
        s_next, r = step(s, a)
        total += prob_a * (r + GAMMA * V_read[s_next])
    return total


def sup_norm_error(V):
    return np.max(np.abs(V - V_true))


N_SWEEPS = 60

# Jacobi Implementation (two array)
def jacobi_policy_evaluation():
    V_old = np.zeros(N_STATES)
    errors = [sup_norm_error(V_old)]
    for _ in range(N_SWEEPS):
        V_new = np.zeros(N_STATES)
        for s in range(N_STATES):
            # every update reads ONLY from V_old -> no in-sweep information sharing
            V_new[s] = bellman_backup(s, V_old)
        V_old = V_new
        errors.append(sup_norm_error(V_old))
    return V_old, errors


# Gauss-Seibel implementation (Single array, in-place, fixed sweep order)
def gauss_seidel_policy_evaluation():
    V = np.zeros(N_STATES)
    errors = [sup_norm_error(V)]
    for _ in range(N_SWEEPS):
        for s in range(N_STATES):
            # reads V in-place: earlier states in this sweep are already fresh
            V[s] = bellman_backup(s, V)
        errors.append(sup_norm_error(V))
    return V, errors

# Asynchronous DP (single array, in-place, random order each sweep)
def async_policy_evaluation(seed=0):
    rng = np.random.default_rng(seed)
    V = np.zeros(N_STATES)
    errors = [sup_norm_error(V)]
    for _ in range(N_SWEEPS):
        order = rng.permutation(N_STATES)  # random order, but every state touched once/sweep
        for s in order:
            V[s] = bellman_backup(s, V)
        errors.append(sup_norm_error(V))
    return V, errors

V_jacobi, err_jacobi = jacobi_policy_evaluation()
V_gs, err_gs = gauss_seidel_policy_evaluation()
V_async, err_async = async_policy_evaluation(seed=42)

print("Max |V_jacobi - V_true| :", np.max(np.abs(V_jacobi - V_true)))
print("Max |V_gs     - V_true| :", np.max(np.abs(V_gs - V_true)))
print("Max |V_async  - V_true| :", np.max(np.abs(V_async - V_true)))


plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(14, 8))

# Convergence curves + final value function heatmaps
ax1 = fig.add_subplot(2, 2, (1, 2))
sweeps = np.arange(N_SWEEPS + 1)
ax1.semilogy(sweeps, err_jacobi, label="Jacobi (two-array)", lw=2, marker="o", markevery=5)
ax1.semilogy(sweeps, err_gs, label="Gauss-Seidel (in-place)", lw=2, marker="s", markevery=5)
ax1.semilogy(sweeps, err_async, label="Asynchronous DP (random order)", lw=2, marker="^", markevery=5)
ax1.axhline(1e-8, color="gray", ls="--", lw=1, label="≈ machine precision")
ax1.set_xlabel("Sweep number (each sweep = N updates)")
ax1.set_ylabel(r"$\|V_k - V_\pi\|_\infty$  (log scale)")
ax1.set_title("All three converge to the SAME unique fixed point $V_\\pi$\n"
              "(Banach contraction guarantee) — only speed differs")
ax1.legend()

def plot_value_grid(ax, V, title):
    im = ax.imshow(V.reshape(GRID_SIZE, GRID_SIZE), cmap="viridis")
    for s in range(N_STATES):
        r, c = divmod(s, GRID_SIZE)
        ax.text(c, r, f"{V[s]:.1f}", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return im

ax2 = fig.add_subplot(2, 3, 4)
plot_value_grid(ax2, V_true, "True $V_\\pi$ (direct solve)")

ax3 = fig.add_subplot(2, 3, 5)
plot_value_grid(ax3, V_gs, f"Gauss-Seidel after {N_SWEEPS} sweeps")

ax4 = fig.add_subplot(2, 3, 6)
plot_value_grid(ax4, V_async, f"Async DP after {N_SWEEPS} sweeps")

plt.tight_layout()
plt.savefig("policy_evaluation_contraction.png", dpi=150)
plt.show()


# Is GS always faster than async DP? => No 
# because async DP is a general framework. Fixed - order GS is literally one specific instance of async DP - the special 
# case where random order is replaced by the same deterministic order every time.
def gs_sweeps_to_converge(order, n, gamma, tol=1e-12, max_sweeps=200):
    """Run GS with a given fixed order; count sweeps until ||V-V_true|| < tol."""
    V = np.zeros(n)
    V_true = np.zeros(n)
    for s in range(n - 2, -1, -1):          # exact value: -1-gamma-...-gamma^(n-2-s)
        V_true[s] = -1 + gamma * V_true[s + 1]
    for sweep in range(1, max_sweeps + 1):
        for s in order:
            if s == n - 1:
                continue
            V[s] = -1 + gamma * V[s + 1]     # bellman_backup, in-place
        if np.max(np.abs(V - V_true)) < tol:
            return sweep
    return None  # did not converge within max_sweeps

n, gamma = 20, 0.9
forward  = list(range(n))
backward = list(range(n - 1, -1, -1))
rng = np.random.default_rng(1)
random_order = rng.permutation(n).tolist()

print("Sweeps to exact convergence:")
print("  forward :", gs_sweeps_to_converge(forward, n, gamma))
print("  backward:", gs_sweeps_to_converge(backward, n, gamma))
print("  random  :", gs_sweeps_to_converge(random_order, n, gamma))
