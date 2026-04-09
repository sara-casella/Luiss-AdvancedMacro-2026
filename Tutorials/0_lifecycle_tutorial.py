#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 8 14:22:03 2026

@author: sara
"""

# %%  Import libraries     

import numpy as np
import matplotlib.pyplot as plt

import lifecycle_utils as utils


# %%  Model Parameters

T       = 61            
rho     = 0.04  
beta    = 1 / (1 + rho)
eis     = 1.1  
r       = 0.01
amin    = 0
theta   = 0.7  
a_0 = 0

# Computational parameters
amax = 20  
n_a = 400

# Income profile

y_profile = np.loadtxt("incprofile.txt")
y_profile = np.concatenate((y_profile, theta * np.ones(T - len(y_profile)) * y_profile[-1]))

# Plot
plt.plot(y_profile)
plt.grid()
plt.show()

# %%  Model solution
a_grid_constrained = utils.discretize_assets_uniform(amax, n_a, amin)


a_policy_constrained, c_policy_constrained, V_constrained = utils.vfi_finite(a_grid_constrained, y_profile, r, beta, eis)


# %% Plots

# Plot value function
plt.plot(a_grid_constrained, V_constrained[0, :], label="V_1(a)")
plt.xlabel("Assets (a)")
plt.ylabel("Value Function (V)")
plt.title("Value Function for First Period")
plt.legend()
plt.grid()
plt.show()

# a'_1(a) and a'_59(a)

plt.plot(a_grid_constrained, a_grid_constrained, 'k--')
plt.plot(a_grid_constrained, a_policy_constrained[0, :], label="a'_1(a)")
plt.plot(a_grid_constrained, a_policy_constrained[59, :], label="a'_59(a)")
plt.xlabel("Assets (a)")
plt.ylabel("Next Period's Assets (a')")
plt.title("Policy Functions for Assets")
plt.legend()
plt.grid()
plt.show()


# %% Compare asset grids

a_grid_uniform = utils.discretize_assets_uniform(100, 50)
a_grid_exponential = utils.discretize_assets_exponential(100, 50)

plt.scatter(a_grid_uniform,a_grid_uniform)
plt.scatter(a_grid_exponential,a_grid_uniform)
plt.show()

# %%
