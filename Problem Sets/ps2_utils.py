#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:22:03 2025

@author: sara
"""

import numpy as np


# MEMO DEBUGGER

# breakpoint(): Insert a break point 
# n(ext): Step over
# s(tep): Step into
# r(eturn): Continue until the current function returns
# c(ontinue): Continue until the next breakpoint is encountered
# unt(il) line_number: Continue until a specific line is encountered. 
# q(uit): Exit debugger



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

def vfi_finite_egm(a_grid, y_profile, r, beta, eis):
        
    '''Value function iteration using endogenous grid method
    
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
    a:    array(T x n_a), asset policy function
    c:    array(T x n_a), consumption policy function
    
    '''
    
    T = len(y_profile)

    # Initialize policy functions
    c = np.zeros((T, len(a_grid)))  # Consumption policy
    a = np.zeros((T, len(a_grid)))  # Savings policy

    # Final period: consume all resources
    a[-1, :] = 0  # No savings nor borrowing in the last period
    c[-1, :] = y_profile[-1] + (1 + r) * a_grid  # Consumption in the final period
    c[-1, c[-1,:]<0] = 0 # Consumption cannot be negative
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    for tau in range(T-2, -1, -1):  # Loop from period T-1 to 0

        # Step 1: Forming Expectations
        c_prime = c[tau+1, :]  # Consumption policy in the next period
        Va = beta*(1 + r) * c_prime ** (-1 / eis)
        
        # Step 2: Finding endogenous consumption and resources
        c_endo =  Va ** (-eis)  # FOC
        m_endo = c_endo + a_grid - y_profile[tau]  # endogenous resources
        
        # breakpoint()
        
        # Step 3: Evaluating in our grids
        a[tau, :] = np.interp((1+r)*a_grid, m_endo, a_grid)  # Current savings
       
        # Step 4: constraints (a>amin, c>0)
        a[tau, :] = np.maximum(a[tau, :], a_grid[0])
        c[tau, :] = y_profile[tau] + (1 + r) * a_grid - a[tau, :]
        c[tau, c[tau,:]<0] = 0 # Consumption cannot be negative
        
    return a, c
  

# %%--------------------------- Asset and consumption life-cycle profile ------------

def simulate_lifecycle(a_policy, c_policy, a_grid, start_asset, T):
    
    
    asset_path = np.zeros(T)
    consumption_path = np.zeros(T)
    
    current_asset = start_asset
    
    
    # Function to find the index of the grid point closest to the given asset level.
    def closest_index(a_value):
        return np.argmin(np.abs(a_grid - a_value))

    # Loop over each decision period t = 0, 1, ..., T-1.
    for t in range(T):
        
        idx = closest_index(current_asset)
        
        # Get consumption at time t from the policy function.
        c = c_policy[t, idx]
        consumption_path[t] = c

        # Get next period's asset level from the policy function.
        next_asset = a_policy[t, idx]
        asset_path[t] = next_asset
        
        # Update the current asset
        current_asset = next_asset
        
    return consumption_path, asset_path