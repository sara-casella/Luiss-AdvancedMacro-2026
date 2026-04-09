#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 8 14:22:03 2026

@author: sara
"""

import numpy as np



# %% Asset grid generation

def discretize_assets_uniform(amax, n_a, amin=0):
    """Create uniform grid between amin and amax
    
        
    Input(s)
    ----------
    amax: integer, grid upper bound
    n_a : integer, number of points in grid
    amin: [optional] integer, grid lower bound

    Output(s)
    ----------
    a_grid : array (n_a), uniform grid
    """

    
    # make uniform grid
    a_grid = np.linspace(amin, amax, n_a)
    a_grid[0] = amin
    
    return a_grid


def discretize_assets_exponential(amax, n_a, amin=0):
    """Create grid between amin and amax, with more points towards amin
    
        
    Input(s)
    ----------
    amax: integer, grid upper bound
    n_a : integer, number of points in grid
    amin: [optional] integer, grid lower bound

    Output(s)
    ----------
    a_grid : array (n_a), double exponential grid
    """

    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = np.log(1 + np.log(1 + amax - amin))
    
    # make uniform grid
    u_grid = np.linspace(0, ubar, n_a)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin + np.exp(np.exp(u_grid) - 1) - 1

# %%--------------------------- Functions for value function iteration ------------

def utility_function(c, eis):
    if eis == 1:
        u = np.log(c)
    else:
        u = c ** (1 - 1/eis) / (1 - 1/eis)
    return u

def vfi_finite(a_grid, y_profile, r, beta, eis):
        
    '''Value function iteration using brute force maximization over the asset grid.
    
    Input(s)
    ----------
    a_grid: array(n_a), assets grid
    y_profile: array(T), income profile
    r:      float, interest rate
    beta:   float, discount factor
    eis:    float, elasticity of intertemporal substitution
   
    tol:   [optional] float, tolerance
    maxit: [optional] integer, maximum number of iterations

    Output(s)
    ----------
    a_prime: array(T x n_a), asset policy function
    c:       array(T x n_a), consumption policy function
    
    '''
    
    T = len(y_profile)

    # Initialize Value and policy functions
    c = np.zeros((T, len(a_grid)))          # Consumption policy
    a_prime = np.zeros((T, len(a_grid)))    # Savings policy
    V = np.zeros((T, len(a_grid)))          # Value function

    # Final period: consume all resources
    a_prime[-1, :] = 0  # No savings nor borrowing in the last period
    c[-1, :] = y_profile[-1] + (1 + r) * a_grid  # Consumption in the final period
    c[-1, c[-1,:]<0] = 0 # Consumption cannot be negative

    # Value in the final period
    V[-1, :] = c[-1, :] ** (1 - 1/eis) / (1 - 1/eis)
    #V[-1,:] = utility_function(c[-1, :])

    for tau in range(T-2, -1, -1):  # Loop from period T-1 to 0
        
        for i, a in enumerate(a_grid):  # Loop over current assets

            # Step 1: Compute value for all possible choices of a' in the grid
            V_candidate = np.zeros(len(a_grid))  # Initialize candidate value function for this state

            for j, a_prime_candidate in enumerate(a_grid):  # Loop over candidate next period's assets

                # Step 1a: Compute current consumption for this candidate choice of a'
                c_candidate = y_profile[tau] + (1 + r) * a - a_prime_candidate
                
                # Step 1b: Compute value of this candidate choice
                if c_candidate <= 0:
                    V_candidate[j] = -np.inf  # Set Value to a very low number
                else:
                    V_candidate[j] = c_candidate ** (1 - 1/eis) / (1 - 1/eis) + beta * V[tau+1, j]
                
            # Step 2: Find optimal choice 
            optimal_index = np.argmax(V_candidate)

            # Step 3: Store optimal choice and value
            a_prime[tau, i] = a_grid[optimal_index]
            c[tau, i] = y_profile[tau] + (1 + r) * a - a_prime[tau, i]
            V[tau, i] = V_candidate[optimal_index]

    return a_prime, c, V

  