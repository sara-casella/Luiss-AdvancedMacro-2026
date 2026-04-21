#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:30:56 2025

@author: sara
"""

import numpy as np
import matplotlib.pyplot as plt

import ps2_utils as utils


# %%  Model Parameters

T       = 61            
rho     = 0.04  
beta    = 1 / (1 + rho)
eis     = 1  
r       = 0.05
amin    = -15
theta   = 0.7  
a_0 = 0

amax = 50  
n_a = 100000

# Income profile

y_profile = np.loadtxt("incprofile.txt")
y_profile = np.concatenate((y_profile, theta * np.ones(T - len(y_profile)) * y_profile[-1]))

# %%  Model solution

a_grid = utils.discretize_assets_uniform(amax, n_a, amin)
a_policy, c_policy = utils.vfi_finite_egm(a_grid, y_profile, r, beta, eis)
consumption_path, asset_path = utils.simulate_lifecycle(a_policy, c_policy, a_grid, a_0, T)

a_grid_constrained = utils.discretize_assets_uniform(amax, n_a, 0)
a_policy_constrained, c_policy_constrained = utils.vfi_finite_egm(a_grid_constrained, y_profile, r, beta, eis)
consumption_path_constrained, asset_path_constrained = utils.simulate_lifecycle(a_policy_constrained, c_policy_constrained, a_grid_constrained, a_0, T)


# %% Plots

# a'_1(a) and a'_60(a)

plt.plot(a_grid, a_grid, 'k--')
plt.plot(a_grid, a_policy[58, :], label="a'_59(a)")
plt.plot(a_grid, a_policy[59, :], label="a'_60(a)")
plt.xlabel("Assets (a)")
plt.ylabel("Next Period's Assets (a')")
plt.title("Policy Functions for Assets")
plt.legend()
plt.grid()
plt.show()

plt.plot(a_grid_constrained, a_grid_constrained, 'k--')
plt.plot(a_grid_constrained, a_policy_constrained[0, :], label="a'_1(a)")
plt.plot(a_grid_constrained, a_policy_constrained[59, :], label="a'_60(a)")
plt.xlabel("Assets (a)")
plt.ylabel("Next Period's Assets (a')")
plt.title("Policy Functions for Assets")
plt.legend()
plt.grid()
plt.show()

# Plot the life-cycle consumption path
plt.figure(figsize=(10, 6))
plt.plot(range(20,T+20), consumption_path, marker='o', label="Consumption profile, amin = -10")
plt.plot(range(20,T+20), y_profile, marker='o', label="Income")
plt.xlabel("Age")
plt.ylabel("Consumption")
plt.title("Life-Cycle Consumption Profile")
plt.grid(True)
plt.legend()
plt.show()

# Plot the life-cycle asset path
plt.figure(figsize=(10, 6))
plt.plot(range(20,T+20), asset_path, marker='o', label="Asset profile")
plt.xlabel("Age")
plt.ylabel("Assets")
plt.title("Life-Cycle Asset profile")
plt.grid(True)
plt.legend()
plt.show()

# Plot the life-cycle consumption path
plt.figure(figsize=(10, 6))
plt.plot(range(20,T+20), consumption_path_constrained, marker='o', label="Consumption profile, amin = 0")
plt.plot(range(20,T+20), y_profile, marker='o', label="Income")
plt.xlabel("Age")
plt.ylabel("Consumption")
plt.title("Life-Cycle Consumption Profile")
plt.grid(True)
plt.legend()
plt.show()

# Plot the life-cycle asset path
plt.figure(figsize=(10, 6))
plt.plot(range(20,T+20), asset_path_constrained, marker='o', label="Asset profile, amin = 0")
plt.xlabel("Age")
plt.ylabel("Assets")
plt.title("Life-Cycle Asset profile")
plt.grid(True)
plt.legend()
plt.show()