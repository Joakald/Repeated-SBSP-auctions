
import numpy as np
from numpy.random import default_rng
from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator
from typing import Callable
from numba import njit, prange
from time import perf_counter
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy

# --- PARAMETERS ---

N_A = 100                       # number of auctions in iterations
N_A_big = 10000                 # number of auctions in final run
N_P = 5                         # number of players (excluding the seller)
N_B = 26                        # number of budgets in grid
N_V = 26                        # number of values in grid
GRID_MIN, GRID_MAX = 0,1        # budget and value grids have the same min and max
DELTAS = [0.7,0.95]             # possible values of delta
BETAS = [1.0]                   # possible values of beta
N_ITERATIONS = 15               # max number of rounds in iteration
DAMPING = 0.3                   # fraction of rival budgets that is preserved when updating 
KS_TOLERANCE = 0.02             # minimum random noise should be 1/sqrt(N_A*N_R)
GET_FULL_B1_STRATEGY = False    # get b1 strategy for entire grid (not necessary and takes a looong time)
USE_PMF_RIVALS = True           # update rival budgets with pmfs for better KS convergence

# don't touch these
N_R = N_P-1
N_BETAS = len(BETAS)
N_DELTAS = len(DELTAS)
N_H2 = 2 # number of possible histories up until period 2
N_H3 = 4
seed = 12342445

# --- FUNCTIONS THAT SOLVE THE MODEL ---

# check equality between floats
def is_close(a, b, tol=1e-10):
    return abs(a - b) < tol

# interpolator factory for Eflow3
def make_Eflow3_interpolator(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    Eflow3: np.ndarray,         # (N_B,N_V)
) -> RegularGridInterpolator:

    grids = (B_grid,V_grid)
    interp = RegularGridInterpolator(
        grids,
        Eflow3,
        bounds_error=False,
        fill_value=None
    )

    return interp 

# interpolator factory for Eflow2
def make_Eflow2_interpolator(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    beta_grid: np.ndarray,      # (N_beta)
    delta_grid: np.ndarray,     # (N_delta)
    Eflow2: np.ndarray,         # (N_B,N_V,N_V,N_BETAS,N_DELTAS)
) -> RegularGridInterpolator:

    grids = (B_grid, V_grid, V_grid, beta_grid, delta_grid)
    interp = RegularGridInterpolator(
        grids,
        Eflow2,
        bounds_error=False,
        fill_value=None
    )

    return interp

# interpolator factory for Eflow3from2
def make_Eflow3from2_interpolator(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    beta_grid: np.ndarray,      # (N_beta)
    delta_grid: np.ndarray,     # (N_delta)
    Eflow3from2: np.ndarray,    # (N_B,N_V,N_V,N_BETAS,N_DELTAS)
) -> RegularGridInterpolator:

    grids = (B_grid, V_grid, V_grid, beta_grid, delta_grid)
    interp = RegularGridInterpolator(
        grids,
        Eflow3from2,
        bounds_error=False,
        fill_value=None
    )

    return interp

# interpolator factory for b1 strategy
def make_b1_interpolator(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    beta_grid: np.ndarray,      # (N_beta)
    delta_grid: np.ndarray,     # (N_delta)
    b1_strategy: np.ndarray,    # (N_B,N_V,N_V,N_V,N_BETAS,N_DELTAS)
) -> RegularGridInterpolator:

    grids = (B_grid, V_grid, V_grid, V_grid, beta_grid, delta_grid)
    interp = RegularGridInterpolator(
        grids,
        b1_strategy,
        bounds_error=False,
        fill_value=None
    )

    return interp

# interpolator factory for b2 strategy
def make_b2_interpolator(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    beta_grid: np.ndarray,      # (N_beta)
    delta_grid: np.ndarray,     # (N_delta)
    b2_strategy: np.ndarray,    # (N_B,N_V,N_V,N_BETAS,N_DELTAS)
) -> RegularGridInterpolator:

    grids = (B_grid, V_grid, V_grid, beta_grid, delta_grid)
    interp = RegularGridInterpolator(
        grids,
        b2_strategy,
        bounds_error=False,
        fill_value=None
    )

    return interp

# ks distance for two distribution functions
def ks_distance(
    sample_a: np.ndarray, # (N_A,)
    sample_b: np.ndarray  # (N_A,)
) -> float:

    a = np.sort(sample_a) 
    b = np.sort(sample_b)
    i = j = 0
    na, nb = len(a), len(b)
    d = 0.0
    while i < na and j < nb:
        if a[i] <= b[j]: 
            i += 1
        else:
            j += 1
        d = max(d, abs(i/na - j/nb))

    return d

def ks_from_pmfs(
    p: np.ndarray, 
    q: np.ndarray
) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.max(np.abs(cdf_p - cdf_q))

# return a sample which is a mix of old and new
def damp_sample(
    old_sample: np.ndarray,     # (N_A,N_P)
    new_sample: np.ndarray,     # (N_A,N_P)
    damping: float,
    rng: np.random.Generator
) -> np.ndarray:                # (N_A,N_P)

    assert 0.0 <= damping <= 1.0

    if damping == 0.0:
        return old_sample
    if damping == 1.0:
        return new_sample

    mask = rng.random(size=old_sample.shape) < damping
    mixed = old_sample.copy()
    mixed[mask] = new_sample[mask]

    return mixed

# set up grids for budget values
def make_grids(
    min_point: float, 
    max_point: float,
    n_budget_points: int, 
    n_value_points: int
) -> tuple[np.ndarray, np.ndarray]: # (N_B,) and (N_V,)

    if n_budget_points < n_value_points:
        raise ValueError('grids error: value grid must not have more points than budget grid')

    budget_grid = np.linspace(min_point,max_point,n_budget_points)
    value_indices = np.round(np.linspace(0, n_budget_points-1, n_value_points)).astype(int)
    value_grid = budget_grid[value_indices]

    return budget_grid, value_grid

def draw_onto_grid(
    n_auctions: int,
    n_players: int,
    grid: np.ndarray, # the grid from which we draw
    rng: np.random.Generator,
    dist: str = "uniform",
    probabilities: np.ndarray | None = None,
    exp_param: float = 1.0 # weights are exp(-lam * grid_value)
) -> np.ndarray: # (N_A,N_P)

    """
    randomly draw budgets/values/betas/deltas
    note: if max draw is 1, exp_param should be something in (1,3)
    """

    grid = np.asarray(grid)
    n_grid = grid.shape[0]

    if dist == "uniform":
        probs = np.full(n_grid, 1.0 / n_grid, dtype=float)

    elif dist == "custom":
        if probabilities is None:
            raise ValueError("probabilities must be provided when dist='custom'")
        probs = np.asarray(probabilities, dtype=float)
        if probs.shape[0] != n_grid:
            raise ValueError("grid and probabilities must have the same length")
        total = probs.sum()
        if total <= 0.0:
            raise ValueError("probabilities must sum to a positive number")
        probs = probs / total

    elif dist == "exp":
        weights = np.exp(-exp_param * grid)
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("exponential weights sum to zero or negative")
        probs = weights / total

    else:
        raise ValueError(f"Unknown distribution type: {dist}")

    size = (n_auctions, n_players)
    draws = rng.choice(grid, size=size, replace=True, p=probs)

    return draws

def m3_auctions(
    B3_winwin_sample: np.ndarray,   # (N_A,N_R)
    B3_winlose_sample: np.ndarray,  # (N_A,N_R)
    B3_losewin_sample: np.ndarray,  # (N_A,N_R)
    B3_loselose_sample: np.ndarray, # (N_A,N_R)
    V3_sample: np.ndarray,          # (N_A,N_R)
    n_histories3: int
) -> tuple[np.ndarray,np.ndarray]:  # two (N_A,N_H3)

    """
    get highest rival bid and number of ties for every auction 
    given sample of budgets and values in period 3
    """

    n_a = B3_winwin_sample.shape[0]
    n_r = B3_winwin_sample.shape[1]
    n_h3 = n_histories3

    V3_rival_sample = V3_sample[:,:-1]

    bids3_winwin = np.minimum(B3_winwin_sample,V3_rival_sample)
    bids3_winlose = np.minimum(B3_winlose_sample,V3_rival_sample)
    bids3_losewin = np.minimum(B3_losewin_sample,V3_rival_sample)
    bids3_loselose = np.minimum(B3_loselose_sample,V3_rival_sample)
    bids3 = np.asarray([bids3_winwin,bids3_winlose,
                        bids3_losewin,bids3_loselose]) # (N_H,N_A,N_R)

    m3 = np.empty((n_a,n_h3),float)
    ties3 = np.empty((n_a,n_h3),int)

    for a in range(n_a):
        for h in range(n_h3):
            m3[a,h] = bids3[h,a,:].max() # max within each history and auction
            ties3[a,h] = np.sum(bids3[h,a,:] == m3[a,h])
    
    return m3, ties3

#@njit(parallel=True)
def Eflow3_grid(
    B3_grid: np.ndarray,    #(N_B,)
    V3_grid: np.ndarray,    #(N_V,)
    m3: np.ndarray,         #(N_A,4)
    ties3: np.ndarray       #(N_A,N_H3)
) -> np.ndarray:            # (N_H3,N_B,N_V)

    """
    get Eflow3(B3,v3) for every (B3,V3) in grid
    given b3 strategy and given a distribution of highest rival bid m3
    """

    n_b = B3_grid.shape[0]
    n_v = V3_grid.shape[0]
    n_a = m3.shape[0]
    n_h3 = m3.shape[1]

    Eflow3 = np.empty((n_h3, n_b, n_v), dtype=np.float64)

    # just to be sure...
    safe_ties3 = ties3.copy()
    safe_ties3[safe_ties3 < 1] = 1
    
    # loop through all (B3,V3) and set the Eflow3 containers
    tol = 1e-12
    for i in prange(n_b):
        B3 = B3_grid[i]
        for j in range(n_v):
            
            V3 = V3_grid[j]
            b3 = B3 if B3 < V3 else V3 # numba-safe min

            Eflow3_sums = np.zeros(n_h3,dtype=np.float64)

            for a in range(n_a):             
                for h in range(n_h3): # loop over all four histories
                    y = m3[a, h]
                    t = safe_ties3[a, h]

                    if y < b3 - tol:
                        prob = 1.0
                    elif abs(y - b3) <= tol:
                        prob = 1.0 / (1.0 + t)
                    else:
                        prob = 0.0

                    Eflow3_sums[h] += (V3 - y) * prob

            for h in range(n_h3):
                Eflow3[h, i, j] = Eflow3_sums[h] / n_a
    
    return Eflow3

def foc2(
    B2: float, 
    V2: float, 
    V3: float,
    beta: float, 
    delta: float,
    y: float,
    Eflow3_interpolator_H2win: Callable,
    Eflow3_interpolator_H2lose: Callable,
    mode: str
) -> float:

    """
    foc of period 2 for given (H2,B2,V2,V3,beta,delta) and bid y and Eflow3(B3,v3)
    note that foc2 should be strictly decreasing in y
    """

    Eflow3_win = Eflow3_interpolator_H2win([B2-y,V3])
    Eflow3_lose = Eflow3_interpolator_H2lose([B2,V3])

    if (mode == "sophisticated" or mode == "naive"):
        return V2-y+beta*delta*(Eflow3_win-Eflow3_lose)
    elif (mode == "commitment"):
        return V2-y+delta*(Eflow3_win-Eflow3_lose)
    else:
        raise ValueError(f"Unknown solution mode: {mode}")

def b2_star_grid_single_state(
    B2: float, 
    V2: float, 
    V3: float,
    beta: float, 
    delta: float,
    B_grid: np.ndarray, # (N_B,) 
    V_grid: np.ndarray, # (N_V,)
    Eflow3_interpolator_H2win: Callable,
    Eflow3_interpolator_H2lose: Callable,
    mode: str
) -> float:
    
    """
    find optimal b2 for a particular (H2,B2,V2,V3,beta,delta) in grid 
    given Eflow3 interpolators for each H3
    """

    b_grid = B_grid
    if B2 <= 0.0:
        return 0.0
    def foc_y(y: float) -> float:
        return foc2(B2, V2, V3, beta, delta, y, 
                    Eflow3_interpolator_H2win,Eflow3_interpolator_H2lose,mode)
    
    # find root
    f_left  = foc_y(0.0)
    f_right = foc_y(B2)
    y_star_cont: float 
    if is_close(f_left,0.0): # root is 0
        y_star_cont = 0.0
    elif is_close(f_right, 0.0): # root is B2
        y_star_cont = B2
    elif f_left * f_right < 0.0: # root is in (0,B2)
        y_star_cont = brentq(foc_y, 0.0, B2)
    elif f_left < 0.0: # root is to the left of (0,B2)
        y_star_cont = 0.0
    else: # root is to the right of (0,B2)
        y_star_cont = B2 
    
    # snap to grid
    y_star_cont = min(max(y_star_cont, 0.0), B2)
    idx = int(np.argmin(np.abs(b_grid - y_star_cont)))
    b2_star = b_grid[idx]

    return b2_star

def b2_star_grid(
    B_grid: np.ndarray, # (N_B,)
    V_grid: np.ndarray, # (N_V,)
    beta_grid: np.ndarray,
    delta_grid: np.ndarray,
    Eflow3_interpolators: np.ndarray, # (N_H3,) one interpolator for each history
    n_histories2: int,
    mode: str       # sophisticated or commitment or naive
) -> np.ndarray:    # (N_H2,N_B,N_V,N_V,N_BETAS,N_DELTAS)
    
    """
    find optimal b2 strategy for all (H2,B2,V2,V3,beta,delta) in grid 
    given Eflow3 interpolators each H3
    """

    n_b = len(B_grid)
    n_v = len(V_grid)
    n_betas = len(beta_grid)
    n_deltas = len(delta_grid)
    b2 = np.empty((n_histories2,n_b,n_v,n_v,n_betas,n_deltas), dtype=float)

    # big loop, hopefully wont take too long...
    # can't use numbda because we pass interpolators
    for i in range(n_b):
        B2 = B_grid[i]
        for j in range(n_v):
            V2 = V_grid[j]
            for k in range(n_v):
                V3 = V_grid[k]
                for l in range(n_betas):
                    beta = beta_grid[l]
                    for m in range(n_deltas):
                        delta = delta_grid[m]

                        # b2 for players who won in period 1
                        b2_win = b2_star_grid_single_state(
                            B2,V2,V3,beta,delta,B_grid,V_grid,
                            Eflow3_interpolators[0],Eflow3_interpolators[1],mode)
                        # b2 for players who lost in period 1
                        b2_lose = b2_star_grid_single_state(
                            B2,V2,V3,beta,delta,B_grid,V_grid,
                            Eflow3_interpolators[2],Eflow3_interpolators[3],mode)

                        b2[0,i,j,k,l,m] = b2_win
                        b2[1,i,j,k,l,m] = b2_lose

    return b2

def m2_auctions(
    B_grid: np.ndarray,             # (N_B,)
    V_grid: np.ndarray,             # (N_V,)
    beta_grid: np.ndarray,          # (N_BETAS,)
    delta_grid: np.ndarray,         # (N_DELTAS,)
    B2_win_sample: np.ndarray,      # (N_A,N_R)
    B2_lose_sample: np.ndarray,     # (N_A,N_R)
    V2_sample: np.ndarray,          # (N_A,N_R)
    V3_sample: np.ndarray,          # (N_A,N_R)
    beta_sample: np.ndarray,        # (N_A,N_R)
    delta_sample: np.ndarray,       # (N_A,N_R)
    b2_strategy: np.ndarray,        # (2,N_B,N_V,N_V,N_BETAS,N_DELTAS)
) -> tuple[np.ndarray, np.ndarray]: # two (N_A,2)
    
    """
    get highest rival bid and number of ties for every auction
    for win/lose in period 1 given sample of budgets and values in period 2
    """

    n_a,n_r = B2_win_sample.shape
    n_h2 = b2_strategy.shape[0] 

    b2_win_sample = np.empty_like(B2_win_sample, dtype=float)
    b2_lose_sample = np.empty_like(B2_lose_sample, dtype=float)
    
    for a in range(n_a):
        for r in range(n_r):

            B2_win = B2_win_sample[a,r]
            B2_lose = B2_lose_sample[a,r]
            V2 = V2_sample[a,r]
            V3 = V3_sample[a,r]
            beta = beta_sample[a,r]
            delta = delta_sample[a,r]

            idx_B2_win = int(np.argmin(np.abs(B_grid - B2_win)))
            idx_B2_lose = int(np.argmin(np.abs(B_grid - B2_lose)))
            idx_V2 = int(np.argmin(np.abs(V_grid - V2)))
            idx_V3 = int(np.argmin(np.abs(V_grid - V3)))
            idx_beta = int(np.argmin(np.abs(beta_grid - beta)))
            idx_delta = int(np.argmin(np.abs(delta_grid - delta)))

            b2_win_sample[a,r] = b2_strategy[
                0,idx_B2_win, idx_V2, idx_V3, idx_beta, idx_delta]
            b2_lose_sample[a,r] = b2_strategy[
                1,idx_B2_lose, idx_V2, idx_V3, idx_beta, idx_delta]

    bids2 = np.asarray([b2_win_sample,b2_lose_sample]) # (N_H,N_A,N_R)
    
    m2 = np.empty((n_a,n_h2),float)
    ties2 = np.empty((n_a,n_h2),int)

    for a in range(n_a):
        for h in range(n_h2):

            m2[a,h] = bids2[h,a,:].max() # max within each history and auction
            ties2[a,h] = np.sum(bids2[h,a,:] == m2[a,h])
    
    return m2, ties2

def build_Eflow3_tables(
    B_grid: np.ndarray, # (N_B,) 
    V_grid: np.ndarray, # (N_V,) 
    m2: np.ndarray,     # (N_A,) 
    Eflow3_interpolator: Callable
) -> tuple[np.ndarray,np.ndarray]: # lose: (N_B,N_V), win: (N_B,N_V,N_A)
    
    """
    get Eflow3 tables for grid (B3,V3) and for win/lose and for every auction
    note: win table is bigger because B3 will depend on m2 which differs over auctions
    """

    n_b = len(B_grid)
    n_v = len(V_grid)
    n_a = len(m2)
    U3_lose = np.empty((n_b,n_v))
    U3_win  = np.empty((n_b,n_v,n_a))

    for i in range(n_b):
        for j in range(n_v):

            U3_lose[i,j] = Eflow3_interpolator((B_grid[i],V_grid[j]))
            for a in range(n_a):
                """
                note: here we also fill elements in the win table that will never be
                realised in practice because the highest rival bid exceeds my budget and
                therefore also my own bid. In principle, we can skip those states, 
                but the gain should be small.
                """
                B3 = B_grid[i] - m2[a]
                U3_win[i,j,a] = Eflow3_interpolator((B3,V_grid[j]))

    return U3_lose, U3_win

#@njit(parallel=True)
def Eflow2_and_Eflow3from2_grid(
    B_grid: np.ndarray,                 # (N_B,)
    V_grid: np.ndarray,                 # (N_V,)
    beta_grid: np.ndarray,              # (N_BETAS,)
    delta_grid: np.ndarray,             # (N_DELTAS,)
    m2: np.ndarray,                     # (N_A,N_H2)
    ties2: np.ndarray,                  # (N_A,N_H2)
    b2_strategy: np.ndarray,            # (N_H2,N_B,N_V,N_V,N_BETA,N_DELTA)
    Eflow3_interpolators: np.ndarray,   # (N_H4,) one interpolator for each H4
) -> tuple[np.ndarray,np.ndarray]:      # two (N_H2,N_B,N_V,N_V,N_BETAS,N_DELTAS)
    
    """
    get Eflow2 and Eflow3from2 for every (B2,v2,v3,beta,delta) in grid
    by using b2 strategy and m2
    """

    n_b = len(B_grid)
    n_v = len(V_grid)
    n_betas = len(beta_grid)
    n_deltas = len(delta_grid)
    n_a,n_h2 = m2.shape

    Eflow2 = np.empty((n_h2,n_b,n_v,n_v,n_betas,n_deltas), dtype=np.float64)
    Eflow3from2 = np.empty((n_h2,n_b,n_v,n_v,n_betas,n_deltas),dtype=np.float64)

    safe_ties2 = ties2.copy()
    safe_ties2[safe_ties2 < 1] = 1
    
    # loop through all (H2,B2,V2,V3,Beta,Delta) 
    # and set the Eflow2 and Eflow3from2 containers
    tol = 1e-12
    for i in prange(n_b):
        B2 = B_grid[i]
        print(B2) # to check where we are while running

        for j in range(n_v):
            V2 = V_grid[j]

            for k in range(n_v):
                V3 = V_grid[k]

                for l in range(n_betas):
                    beta = beta_grid[l]

                    for m in range(n_deltas):
                        delta = delta_grid[m]  

                        Eflow2_win_sum = 0.0  
                        Eflow2_lose_sum = 0.0
                        Eflow3from2_win_sum = 0.0
                        Eflow3from2_lose_sum = 0.0

                        b2_win = b2_strategy[0,i,j,k,l,m]
                        b2_lose = b2_strategy[1,i,j,k,l,m]

                        for a in range(N_A):
                            
                            # win in period 1
                            if m2[a,0] < b2_win - tol:
                                prob = 1.0
                            elif abs(m2[a,0] - b2_win) <= tol:
                                prob = 1.0 / (1.0 + safe_ties2[a,0])
                            else:
                                prob = 0.0
                            Eflow2_win_sum += (V2 - m2[a,0]) * prob
                            Eflow3from2_winwin = Eflow3_interpolators[0]([B2-b2_win,V3])
                            Eflow3from2_winlose = Eflow3_interpolators[1]([B2,V3])
                            Eflow3from2_win_sum += Eflow3from2_winwin * prob \
                            + Eflow3from2_winlose * (1.0 - prob)

                            # lose in period 1
                            if m2[a,1] < b2_lose - tol:
                                prob = 1.0
                            elif abs(m2[a,1] - b2_lose) <= tol:
                                prob = 1.0 / (1.0 + safe_ties2[a,1])
                            else:
                                prob = 0.0
                            Eflow2_lose_sum += (V2 - m2[a,1]) * prob
                            Eflow3from2_losewin = Eflow3_interpolators[2]([B2-b2_lose,V3])
                            Eflow3from2_loselose = Eflow3_interpolators[3]([B2,V3])
                            Eflow3from2_lose_sum += Eflow3from2_losewin * prob \
                            + Eflow3from2_loselose * (1.0 - prob)
                      
                        Eflow2[0,i,j,k,l,m] = Eflow2_win_sum / n_a
                        Eflow2[1,i,j,k,l,m] = Eflow2_lose_sum / n_a
                        Eflow3from2[0,i,j,k,l,m] = np.asarray(Eflow3from2_win_sum / n_a).item()
                        Eflow3from2[1,i,j,k,l,m] = np.asarray(Eflow3from2_lose_sum / n_a).item()

    return Eflow2, Eflow3from2

def foc1(B1: float, 
    v1: float, 
    v2: float, 
    v3: float,
    beta: float, 
    delta: float, 
    y: float,
    Eflow2_interpolators: np.ndarray,       # (N_H2,) 
    Eflow3from2_interpolators: np.ndarray,  # (N_H2,)
) -> float:

    """
    foc of period 1 for given (B1,v1,v2,v3,beta,delta) in grid 
    and given bid y and Eflow2 and Eflow3from2 interpolators
    """

    CV_win = Eflow2_interpolators[0]([B1-y,v2,v3,beta,delta]) + \
        delta * Eflow3from2_interpolators[0]([B1-y,v2,v3,beta,delta])

    CV_lose = Eflow2_interpolators[1]([B1,v2,v3,beta,delta]) + \
        delta * Eflow3from2_interpolators[1]([B1,v2,v3,beta,delta])

    return v1-y+beta*delta*(CV_win-CV_lose)

def b1_star_grid_single_state(
    B1: float, 
    V1: float, 
    V2: float, 
    V3: float,
    beta: float, 
    delta: float,
    B_grid: np.ndarray,                     # (N_B,) 
    V_grid: np.ndarray,                     # (N_V,)
    Eflow2_interpolators: np.ndarray,       # (N_H2,)
    Eflow3from2_interpolators: np.ndarray   # (N_H2,) 
) -> float:
    
    """
    find optimal b1 strategy for a particular (B1,V1,V2,V3,beta,delta) in grid 
    given Eflow3 and Eflow3from2 interpolators
    """

    b_grid = B_grid
    if B1 <= 0.0:
        return 0.0
    def foc_y(y: float) -> float:
        return foc1(B1,V1, V2, V3,beta, delta,y,
                   Eflow2_interpolators,Eflow3from2_interpolators)

    # find root
    f_left  = foc_y(0.0)
    f_right = foc_y(B1)
    y_star_cont: float 
    if is_close(f_left,0.0): # root is 0
        y_star_cont = 0.0
    elif is_close(f_right, 0.0): # root is B1
        y_star_cont = B1
    elif f_left * f_right < 0.0: # root is in (0,B1)
        y_star_cont = brentq(foc_y, 0.0, B1)
    elif f_left < 0.0: # root is to the left of (0,B1)
            y_star_cont = 0.0
    else: # root is to the right of (0,B1)
        y_star_cont = B1 
    
    # snap to grid
    y_star_cont = min(max(y_star_cont, 0.0), B1)
    idx = int(np.argmin(np.abs(b_grid - y_star_cont)))
    while b_grid[idx] > B1 and idx > 0:
        idx -= 1

    b1_star = b_grid[idx]

    return b1_star

def b1_star_grid(
    B_grid: np.ndarray,                     # (N_B,)
    V_grid: np.ndarray,                     # (N_V,)
    beta_grid: np.ndarray,                  # (N_BETAS,)
    delta_grid: np.ndarray,                 # (N_DELTAS,)
    Eflow2_interpolators: np.ndarray,       # (N_H2,)
    Eflow3from2_interpolators: np.ndarray   # (N_H2,)
    ) -> np.ndarray:                        # (N_B,N_V,N_V,N_V,N_beta,N_delta)

    """
    find optimal b1 for every (B1,V1,V2,V3,beta,delta) in grid 
    given Eflow2 and Eflow3from2 interpolators
    """

    n_b     = len(B_grid)
    n_v     = len(V_grid)
    n_betas  = len(beta_grid)
    n_deltas = len(delta_grid)
    b1 = np.empty((n_b,n_v,n_v,n_v,n_betas,n_deltas), dtype=float)
    b_grid = B_grid

    # begin crazy loop...
    for i in range(n_b):
        B1 = B_grid[i]
        print(B1)

        for j in range(n_v):
            V1 = V_grid[j]

            for k in range(n_v):
                V2 = V_grid[k]

                for l in range(n_v):
                    V3 = V_grid[l]

                    for m in range(n_betas):
                        beta = beta_grid[m]

                        for n in range(n_deltas):
                            delta = delta_grid[n]

                            b1[i,j,k,l,m,n] = b1_star_grid_single_state(
                                B1,V1,V2,V3,beta,delta,
                                B_grid,V_grid,
                                Eflow2_interpolators,Eflow3from2_interpolators)
                            
    return b1

def b1_star_auctions(
    B_grid: np.ndarray,                     # (N_B,)
    V_grid: np.ndarray,                     # (N_V,)
    B1_sample: np.ndarray,                  # (N_A, N_P)
    V1_sample: np.ndarray,                  # (N_A, N_P)
    V2_sample: np.ndarray,                  # (N_A, N_P)
    V3_sample: np.ndarray,                  # (N_A, N_P)
    beta_sample: np.ndarray,                # (N_A, N_P)
    delta_sample: np.ndarray,               # (N_A, N_P)
    Eflow2_interpolators: np.ndarray,       # (N_H2,)
    Eflow3from2_interpolators: np.ndarray   # (N_H2,)
) -> np.ndarray:                            # (N_A, N_P)

    """
    find optimal b1 for auction and player given in a sample
    of (B1,V1,V2,V3,beta,delta) and Eflow2 and Eflow3from2 interpolators
    """

    n_a,n_p = B1_sample.shape
    b1 = np.empty((n_a,n_p), dtype=float)

    for a in range(n_a):
        for p in range(n_p):

            B1    = B1_sample[a, p]
            V1    = V1_sample[a, p]
            V2    = V2_sample[a, p]
            V3    = V3_sample[a, p]
            beta  = beta_sample[a, p]
            delta = delta_sample[a, p]

            # Solve using the single-state solver
            b1[a,p] = b1_star_grid_single_state(
                B1, V1, V2, V3, beta, delta,
                B_grid, V_grid,
                Eflow2_interpolators, Eflow3from2_interpolators)

    return b1

def update_budget(
    bid_sample: np.ndarray,     # (N_A, N_P)
    budget_sample: np.ndarray,  # (N_A, N_P)
    rng: np.random.Generator
) -> np.ndarray:                # one (N_A, N_P) and one (N_A,)

    """
    take the budgets from previous period along with the bids 
    and substracts the transfers and return the new budgets for
    the next period
    """

    n_a = bid_sample.shape[0]

    # get winner in each auction
    max_bids = bid_sample.max(axis=1)
    is_winner_mask = (bid_sample == max_bids[:, None]) # (N_A, N_P) bool
    winners = np.empty(n_a, dtype=np.int64)
    for i in range(n_a):
        tied = np.flatnonzero(is_winner_mask[i]) # just the winners
        winners[i] = rng.choice(tied)
    
    # find payments in each auction and adjust budgets
    payments = np.partition(bid_sample, -2, axis=1)[:, -2]
    rows = np.arange(n_a)
    budget_sample[rows, winners] -= payments

    return budget_sample, winners

def draw_from_budget_sample(
    budgets_sample: np.ndarray, # (x, N_R) where x <= N_A
    n_auctions: int,            # how many rows we want to have in the return object
    rival_draws: np.ndarray,    # (N_A, N_R)
    rng: np.random.Generator
) -> np.ndarray:                # (N_A, N_R)

    """
    Take the rival budgets from the simulation under some given history
    and randomly draw from that to get rival budgets for the next simulation.
    """

    budgets_sample = np.asarray(budgets_sample)

    n_a = n_auctions
    n_rows_sample, n_r = budgets_sample.shape

    if n_rows_sample == 0:
        raise ValueError("Cannot resample: no observations in budgets_sample.")

    flat = budgets_sample.ravel()
    flat_rival_draws = rival_draws.ravel()

    idx = (flat_rival_draws * flat.size).astype(int)
    # just in case floating point gives flat.size occasionally:
    idx = np.minimum(idx, flat.size - 1)

    return flat[idx].reshape(n_a, n_r)

def build_rival_budgets_histories(
    B3_full: np.ndarray,        # (N_A, N_P)                                  
    winners1: np.ndarray,       # (N_A,)
    winners2: np.ndarray,       # (N_A,)
    rng: np.random.Generator
) -> np.ndarray:                # 3 (x,N_R) where x<=N_A  

    """
    Get rival budgets in period 3 for each history except winwin
    given full budgets in period 3. We don't do winwin because
    there's no point in updating it.
    """

    B3_full = np.asarray(B3_full)
    w1 = np.asarray(winners1).ravel()
    w2 = np.asarray(winners2).ravel()

    n_a,n_p = B3_full.shape

    players = np.arange(n_p)

    rows_LL = []
    rows_LW = []
    rows_WL = []

    for i in range(n_a):

        # lost both
        LL_mask = (players != w1[i]) & (players != w2[i])
        LL_candidates = players[LL_mask]
        if LL_candidates.size > 0:
            focal_player = rng.choice(LL_candidates)
            rows_LL.append(np.delete(B3_full[i], focal_player))

        # won first, lost second
        WL_mask = (players == w1[i]) & (players != w2[i])
        WL_candidates = players[WL_mask]
        if WL_candidates.size > 0:
            if WL_candidates.size == 1:
                focal_player = WL_candidates[0]
            else:
                focal_player = rng.choice(WL_candidates)
            rows_WL.append(np.delete(B3_full[i], focal_player))

        # lost first, won second
        LW_mask = (players != w1[i]) & (players == w2[i])
        LW_candidates = players[LW_mask]
        if LW_candidates.size > 0:
            if LW_candidates.size == 1:
                focal_player = LW_candidates[0]
            else: 
                focal_player = rng.choice(LW_candidates)
            rows_LW.append(np.delete(B3_full[i], focal_player))

    def _empty():
        return np.empty((0, n_p - 1))

    B3_LL = np.vstack(rows_LL) if rows_LL else _empty()
    B3_LW = np.vstack(rows_LW) if rows_LW else _empty()
    B3_WL = np.vstack(rows_WL) if rows_WL else _empty()

    return B3_WL, B3_LW, B3_LL

def forward_simulation(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    beta_grid: np.ndarray,      # (N_BETAS,)
    delta_grid: np.ndarray,     # (N_DELTAS,)
    B1_sample: np.ndarray,      # (N_A, N_P)
    V1_sample: np.ndarray,      # (N_A, N_P)
    V2_sample: np.ndarray,      # (N_A, N_P)
    V3_sample: np.ndarray,      # (N_A, N_P)
    beta_sample: np.ndarray,    # (N_A, N_P)
    delta_sample: np.ndarray,   # (N_A, N_P)
    b1_sample: np.ndarray,      # (N_A, N_P)
    b2_strategy: np.ndarray,    # (N_H2, N_B, N_V, N_V, N_BETAS, N_DELTAS)
    rival_draws: np.ndarray,    # (N_A, N_R)
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # 5 (N_A,N_P) and 4 (N_A,N_R)

    """
    Get outcome for bids and budgets of the forward simulation
    and also the distributions of the budgets of rival under the histories
    """

    n_a,n_p = B1_sample.shape
    n_r = n_p - 1

    b2_sample = np.empty_like(B1_sample)
    b3_sample = np.empty_like(B1_sample)
    
    # solve auctions in period 1
    B2_sample, winners1 = update_budget(b1_sample,B1_sample.copy(),rng)

    # get new period 2 rival budget distribution for lose
    # (for win it's the same as in period 1 so we don't update it)
    mask = np.ones_like(B2_sample, dtype=bool)
    rows = np.arange(n_a)
    mask[rows, winners1] = False
    new_B2_rival_lose = B2_sample[mask].reshape(n_a, n_r)

    # get pmf for next iteration
    _ , pmf_lose = _empirical_pmf(new_B2_rival_lose,B_grid)

    # fill b2 sample and get B3
    b2_win_interpolator = make_b2_interpolator(
        B_grid,V_grid,beta_grid,delta_grid,b2_strategy[0,:,:,:,:,:])
    b2_lose_interpolator = make_b2_interpolator(
        B_grid,V_grid,beta_grid,delta_grid,b2_strategy[1,:,:,:,:,:])
    for a in range(n_a):
        for p in range(n_p):
            B2 = B2_sample[a,p]
            V2 = V2_sample[a,p]
            V3 = V3_sample[a,p]
            beta = beta_sample[a,p]
            delta = delta_sample[a,p]
            if winners1[a] == p:
                b2_sample[a,p] = b2_win_interpolator((B2,V2,V3,beta,delta))
            else:
                b2_sample[a,p] = b2_lose_interpolator((B2,V2,V3,beta,delta))
    B3_sample, winners2 = update_budget(
        b2_sample,B2_sample.copy(),rng)

    # get new period 3 rival budget distributions for winlose, losewin and loselose
    # (for winwin it's the same as in period 1 so we don't update it)
    winlose, losewin, loselose = build_rival_budgets_histories(
        B3_sample,winners1,winners2,rng)

    # in the VERY rare case that the matrices have no rows, use B3_sample as a fallback
    def ensure_nonempty(sample: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if sample.shape[0] == 0:
            return fallback
        return sample
    B3_rival_uncond = B3_sample[:, :-1]
    winlose = ensure_nonempty(winlose, B3_rival_uncond)
    losewin = ensure_nonempty(losewin, B3_rival_uncond)
    loselose = ensure_nonempty(loselose, B3_rival_uncond)

    # build new rival budgets for next iteration
    new_B3_rival_winlose = draw_from_budget_sample(winlose,n_a,rival_draws,rng)
    new_B3_rival_losewin = draw_from_budget_sample(losewin,n_a,rival_draws,rng)
    new_B3_rival_loselose = draw_from_budget_sample(loselose,n_a,rival_draws,rng)

    # create pmfs for the next iteration
    winlose_data_for_pmf = winlose.ravel()
    losewin_data_for_pmf = losewin.ravel()
    loselose_data_for_pmf = loselose.ravel()
    _ , pmf_winlose = _empirical_pmf(winlose_data_for_pmf, B_grid)
    _ , pmf_losewin = _empirical_pmf(losewin_data_for_pmf, B_grid)
    _ , pmf_loselose = _empirical_pmf(loselose_data_for_pmf, B_grid)

    # fill b3 sample
    for a in range(n_a):
        for p in range(n_p):
            B3 = B3_sample[a,p]
            V3 = V3_sample[a,p]
            b3_sample[a,p] = np.minimum(B3,V3)

    return b1_sample, b2_sample, b3_sample, \
        B2_sample, B3_sample, \
        new_B2_rival_lose, new_B3_rival_winlose, \
        new_B3_rival_losewin, new_B3_rival_loselose, \
        pmf_lose, pmf_winlose, pmf_losewin, pmf_loselose

def three_period_solver(
    B_grid: np.ndarray,         # (N_B,)
    V_grid: np.ndarray,         # (N_V,)
    beta_grid: np.ndarray,      # (N_BETAS,)
    delta_grid: np.ndarray,     # (N_DELTAS,)
    B1_sample: np.ndarray,      # (N_A, N_P)
    V1_sample: np.ndarray,      # (N_A, N_P)
    V2_sample: np.ndarray,      # (N_A, N_P)
    V3_sample: np.ndarray,      # (N_A, N_P)
    beta_sample: np.ndarray,    # (N_A, N_P)
    delta_sample: np.ndarray,   # (N_A, N_P)
    rng: np.random.Generator,
    ks_tolerance: float,
    n_iterations: int,
    damping: float,
    n_histories2: int,
    n_histories3: int,
    use_pmf_rivals: bool,
    mode: str                   # sophisticated or commitment or naive
) -> dict:                      # dictionary with all the returned 2d-arrays

    """
    solve the three periods auctions and return the outcome for strategies, bids and budgets,
    and the distributions of the rival bids under the different histories
    """

    n_b = len(B_grid)
    n_v = len(V_grid)
    n_betas = len(beta_grid)
    n_deltas = len(delta_grid)
    n_a,n_p = B1_sample.shape
    n_r = n_p -1
    n_h2 = n_histories2
    n_h3 = n_histories3
    
    # declare containers before the loop because I'm a C++ kind of person :P
    b1_strategy_sample = None
    b1_strategy = None
    b2_strategy = None

    # budgets of rivals
    B2_rival_win = B1_sample.copy()
    B2_rival_lose = B1_sample.copy()
    B3_rival_winwin = B1_sample.copy()
    B3_rival_winlose = B1_sample.copy()
    B3_rival_losewin = B1_sample.copy() 
    B3_rival_loselose = B1_sample.copy()
    
    # remove focal player
    B2_rival_win = B2_rival_win[:,:-1]
    B2_rival_lose = B2_rival_lose[:,:-1]
    B3_rival_winwin = B3_rival_winwin[:,:-1]
    B3_rival_winlose = B3_rival_winlose[:,:-1]    
    B3_rival_losewin = B3_rival_losewin[:,:-1]
    B3_rival_loselose = B3_rival_loselose[:,:-1]

    # pmfs for rivals
    pmf_rivals_lose = np.ones(n_b) / n_b
    pmf_rivals_winlose = np.ones(n_b) / n_b
    pmf_rivals_losewin = np.ones(n_b) / n_b
    pmf_rivals_loselose = np.ones(n_b) / n_b

    # KS distances
    KS2_lose_arr = []
    KS3_winlose_arr = []
    KS3_losewin_arr = []
    KS3_loselose_arr = []
    
    # containers for Eflow2 and Eflow3from2 interpolators
    # (we need them to get b1 strategy after the iteration)
    Eflow2_interpolators_final = None
    Eflow3from2_interpolators_final = None

    # set the indices for drawing when updating rival budgets
    rival_draws = rng.random(size=(n_a, n_r))

    print(f"\n MODE: {mode}")

    for it in range(n_iterations):

        print(f"\n=== Iteration {it} ===")

        # === Period 3 ===
        t3 = perf_counter()
        m3, ties3 = m3_auctions(B3_rival_winwin, B3_rival_winlose,
            B3_rival_losewin, B3_rival_loselose,V3_sample,n_h3)
        Eflow3 = Eflow3_grid(B_grid,V_grid,m3,ties3)
        Eflow3_interpolators = []
        for h in range(n_h3):
            interpolator = make_Eflow3_interpolator(B_grid, V_grid, Eflow3[h])
            Eflow3_interpolators.append(interpolator)
        print("  Period-3 all:", perf_counter() - t3, "seconds")

        # === Period 2 ===
        t2a = perf_counter()

        b2_strategy_sophisticated = b2_star_grid(B_grid,V_grid,beta_grid,delta_grid,
            Eflow3_interpolators,n_h2,"sophisticated")
        b2_strategy_commitment = b2_star_grid(B_grid,V_grid,beta_grid,delta_grid,
            Eflow3_interpolators,n_h2,"commitment")

        print("  Period-2 strategies:", perf_counter() - t2a, "seconds")
        
        if (mode == "sophisticated"):
            t2b = perf_counter()
            m2, ties2 = m2_auctions(
                B_grid,V_grid,beta_grid,delta_grid,B2_rival_win,B2_rival_lose,
                V2_sample,V3_sample,beta_sample,delta_sample,b2_strategy_sophisticated)
            print("  Period-2 m2:", perf_counter() - t2b, "seconds")
            t2c = perf_counter()
            Eflow2, Eflow3from2 = Eflow2_and_Eflow3from2_grid(
                B_grid,V_grid,beta_grid,delta_grid,m2,ties2,
                b2_strategy_sophisticated,Eflow3_interpolators)
            new_b2_strategy = b2_strategy_sophisticated
            print("  Period-2 flow:", perf_counter() - t2c, "seconds")
        
        elif (mode == "commitment"):
            t2b = perf_counter()
            m2, ties2 = m2_auctions(
                B_grid,V_grid,beta_grid,delta_grid,B2_rival_win,B2_rival_lose,
                V2_sample,V3_sample,beta_sample,delta_sample,b2_strategy_commitment)
            print("  Period-2 m2:", perf_counter() - t2b, "seconds")
            t2c = perf_counter()
            Eflow2, Eflow3from2 = Eflow2_and_Eflow3from2_grid(
                B_grid,V_grid,beta_grid,delta_grid,m2,ties2,
                b2_strategy_commitment,Eflow3_interpolators)
            new_b2_strategy = b2_strategy_commitment
            print("  Period-2 flow:", perf_counter() - t2c, "seconds")

        elif (mode == "naive"):
            t2b = perf_counter()
            m2, ties2 = m2_auctions(
                B_grid,V_grid,beta_grid,delta_grid,B2_rival_win,B2_rival_lose,
                V2_sample,V3_sample,beta_sample,delta_sample,b2_strategy_commitment)
            print("  Period-2 m2:", perf_counter() - t2b, "seconds")
            t2c = perf_counter()
            Eflow2, Eflow3from2 = Eflow2_and_Eflow3from2_grid(
                B_grid,V_grid,beta_grid,delta_grid,m2,ties2,
                b2_strategy_commitment,Eflow3_interpolators)
            new_b2_strategy = b2_strategy_sophisticated
            print("  Period-2 flow:", perf_counter() - t2c, "seconds")

        else:
            raise ValueError(f"Unknown solution mode: {mode}")

        t2d = perf_counter()
        Eflow2_interpolators = []
        Eflow3from2_interpolators = []
        for h in range(n_h2):
            Eflow2_interpolator = make_Eflow2_interpolator(
                B_grid, V_grid, beta_grid, delta_grid, Eflow2[h])
            Eflow2_interpolators.append(Eflow2_interpolator)
            Eflow3from2_interpolator = make_Eflow3from2_interpolator(
                B_grid, V_grid, beta_grid, delta_grid, Eflow3from2[h])
            Eflow3from2_interpolators.append(Eflow3from2_interpolator)
        print("  Period-2 interpolators:", perf_counter() - t2d, "seconds")

        # === Period 1 ===
        t1 = perf_counter()
        new_b1_strategy_sample = b1_star_auctions(
            B_grid,V_grid,
            B1_sample,V1_sample,V2_sample,V3_sample,beta_sample,delta_sample,
            Eflow2_interpolators,Eflow3from2_interpolators)
        print("  Period-1 all:", perf_counter() - t1, "seconds")

        # === Forward simulation ===
        tf = perf_counter()
        b1_sample, b2_sample, b3_sample, B2_sample, B3_sample, \
            new_B2_rival_lose, new_B3_rival_winlose, \
            new_B3_rival_losewin, new_B3_rival_loselose, \
            new_pmf_lose, new_pmf_winlose, new_pmf_losewin, new_pmf_loselose = forward_simulation(
            B_grid,V_grid,beta_grid,delta_grid,B1_sample.copy(),V1_sample,
            V2_sample,V3_sample,beta_sample,delta_sample,
            new_b1_strategy_sample,new_b2_strategy,rival_draws,rng)
        print("  Forward simulation:", perf_counter() - tf, "seconds")

        # === KS values and update ===
        if(use_pmf_rivals):

            KS2_lose = ks_from_pmfs(pmf_rivals_lose, new_pmf_lose)
            KS3_winlose = ks_from_pmfs(pmf_rivals_winlose, new_pmf_winlose)
            KS3_losewin = ks_from_pmfs(pmf_rivals_losewin, new_pmf_losewin)
            KS3_loselose = ks_from_pmfs(pmf_rivals_loselose, new_pmf_loselose)

            KS2_lose_arr.append(KS2_lose)
            KS3_winlose_arr.append(KS3_winlose)
            KS3_losewin_arr.append(KS3_losewin)
            KS3_loselose_arr.append(KS3_loselose)
            
            # damp when updating
            pmf_rivals_lose     = (1.0 - damping) * pmf_rivals_lose \
                + damping * np.asarray(new_pmf_lose, float)
            pmf_rivals_winlose  = (1.0 - damping) * pmf_rivals_winlose \
                + damping * np.asarray(new_pmf_winlose, float)
            pmf_rivals_losewin  = (1.0 - damping) * pmf_rivals_losewin \
                + damping * np.asarray(new_pmf_losewin, float)
            pmf_rivals_loselose = (1.0 - damping) * pmf_rivals_loselose \
                + damping * np.asarray(new_pmf_loselose, float)

            # normalize just in case
            pmf_rivals_lose /= pmf_rivals_lose.sum()
            pmf_rivals_winlose /= pmf_rivals_winlose.sum()
            pmf_rivals_losewin /= pmf_rivals_losewin.sum()
            pmf_rivals_loselose /= pmf_rivals_loselose.sum()

            # draw new budgets and update
            draws_lose = rng.choice(B_grid, size=n_a * n_r, replace=True, p=pmf_rivals_lose)
            B2_rival_lose = draws_lose.reshape(n_a, n_r)

            draws_winlose = rng.choice(B_grid, size=n_a * n_r, replace=True, p=pmf_rivals_winlose)
            B3_rival_winlose = draws_winlose.reshape(n_a, n_r)

            draws_losewin = rng.choice(B_grid, size=n_a * n_r, replace=True, p=pmf_rivals_losewin)
            B3_rival_losewin = draws_losewin.reshape(n_a, n_r)

            draws_loselose = rng.choice(B_grid, size=n_a * n_r, replace=True, p=pmf_rivals_loselose)
            B3_rival_loselose = draws_loselose.reshape(n_a, n_r)

        else:

            KS2_lose = ks_distance(B2_rival_lose.flatten(), new_B2_rival_lose.flatten())
            KS3_winlose = ks_distance(B3_rival_winlose.flatten(), new_B3_rival_winlose.flatten())
            KS3_losewin = ks_distance(B3_rival_losewin.flatten(), new_B3_rival_losewin.flatten())
            KS3_loselose = ks_distance(B3_rival_loselose.flatten(), new_B3_rival_loselose.flatten())
            
            B2_rival_lose = damp_sample(B2_rival_lose,new_B2_rival_lose,damping,rng)
            B3_rival_winlose = damp_sample(B3_rival_winlose,new_B3_rival_winlose,damping,rng)
            B3_rival_losewin = damp_sample(B3_rival_losewin,new_B3_rival_losewin,damping,rng)
            B3_rival_loselose = damp_sample(B3_rival_loselose,new_B3_rival_loselose,damping,rng)
        
        if(b2_strategy is not None):
            same_b2 = np.isclose(b2_strategy, new_b2_strategy, atol=1e-5)
            changed_b2 = ~same_b2
            frac_changed_b2 = np.mean(changed_b2)
        else:
            frac_changed_b2 = np.inf
        b2_strategy = new_b2_strategy

        if(b1_strategy_sample is not None):
            same_b1 = np.isclose(b1_strategy_sample, new_b1_strategy_sample, atol=1e-5)
            changed_b1 = ~same_b1
            frac_changed_b1 = np.mean(changed_b1)
        else:
            frac_changed_b1 = np.inf
        b1_strategy_sample = new_b1_strategy_sample
            
        print(f"b1 changed fraction = {frac_changed_b1}, \
            b2 changed fraction = {frac_changed_b2}")
        print(f"Iter {it}: KS2_lose={KS2_lose:.4f}, KS3_winlose={KS3_winlose:.4f},\
            KS3_losewin={KS3_losewin:.4f}, KS3_loselose={KS3_loselose:.4f}")

        if (KS2_lose < ks_tolerance) and (KS3_winlose < ks_tolerance) \
            and (KS3_losewin < ks_tolerance) and (KS3_loselose < ks_tolerance):
            Eflow2_interpolators_final = Eflow2_interpolators
            Eflow3from2_interpolators_final = Eflow3from2_interpolators
            break
    
    if Eflow2_interpolators_final is None:
        Eflow2_interpolators_final = Eflow2_interpolators
        Eflow3from2_interpolators_final = Eflow3from2_interpolators

    for i in range(len(KS2_lose_arr)):
        print(f"\n{KS2_lose_arr[i]} {KS3_winlose_arr[i]} "
                f"{KS3_losewin_arr[i]} {KS3_loselose_arr[i]}")

    result = {
        "b2_strategy": b2_strategy,
        "b1_sample": b1_sample,
        "b2_sample": b2_sample,
        "b3_sample": b3_sample,
        "B1_sample": B1_sample,
        "B2_sample": B2_sample,
        "B3_sample": B3_sample,
        "B2_rival_dist_lose": B2_rival_lose,
        "B3_rival_dist_winlose": B3_rival_winlose,
        "B3_rival_dist_losewin": B3_rival_losewin,
        "B3_rival_dist_loselose": B3_rival_loselose,
        "Eflow2_interpolators": Eflow2_interpolators_final,
        "Eflow3from2_interpolators": Eflow3from2_interpolators_final,
        }
    
    return result

# --- FUNCTIONS FOR OUTPUTTING RESULTS ---

def realized_winners(
    bids: np.ndarray,           # (N_A, N_P)
    rng: np.random.Generator
) -> np.ndarray:                # (N_A,)
    """
    Draw a single realized winner per auction, with uniform random tie-breaking.
    """
    bids = np.asarray(bids)
    n_a, _ = bids.shape
    winners = np.empty(n_a, dtype=int)

    for a in range(n_a):
        max_bid = bids[a].max()
        tied = np.flatnonzero(bids[a] == max_bid)
        winners[a] = rng.choice(tied)

    return winners

def _period_benefit_for_group(
    bids: np.ndarray,                   # (N_A, N_P)
    values: np.ndarray,                 # (N_A, N_P)
    mask: np.ndarray,                   # (N_A, N_P) bool: True if player is in beta group
    zero_if_not_in_group: bool = True,
) -> np.ndarray:                        # (N_A,)
    """
    Per auction benefit attributed to the group:
      - if winner is in group: winner_value - second_highest_bid
      - else: 0 (or NaN if zero_if_not_in_group=False)
    """
    bids = np.asarray(bids)
    values = np.asarray(values)
    mask = np.asarray(mask)

    if bids.shape != values.shape or bids.shape != mask.shape:
        raise ValueError("bids, values, mask must have same shape (N_A, N_P)")

    n_a, n_p = bids.shape
    if n_p < 2:
        raise ValueError("Need at least 2 players")

    order = np.argsort(bids, axis=1)
    winner_idx = order[:, -1]
    second_idx = order[:, -2]

    price = bids[np.arange(n_a), second_idx]
    vwin  = values[np.arange(n_a), winner_idx]
    in_group = mask[np.arange(n_a), winner_idx]

    ben = vwin - price
    if zero_if_not_in_group:
        return np.where(in_group, ben, 0.0)
    else:
        return np.where(in_group, ben, np.nan)

def summarize_outcomes(
    result: dict,
    V1_sample: np.ndarray,          # (N_A, N_P)
    V2_sample: np.ndarray,          # (N_A, N_P)
    V3_sample: np.ndarray,          # (N_A, N_P)
    mask: np.ndarray | None = None, # (N_A, N_P) boolean, optional
) -> pd.DataFrame:
    """
    Numeric summaries (mean, std) for:
      - bids in each period (all bids)
      - budgets in each period (all budgets)
      - winning bids in each period, max per auction (or highest bid in the group if we use mask)
      - winner benefits in each period (value - 2nd highest bid) if the winner is in the mask group
      - rival budgets for each history
    """

    def stats(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x).ravel()
        x = x[~np.isnan(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(x)), float(np.std(x))

    rows = []

    # budget
    for t in (1, 2, 3):
        B = np.asarray(result[f"B{t}_sample"])
        if mask is not None:
            B = B[mask]
        mu, sd = stats(B)
        rows.append({
            "category": "budget",
            "detail": "all" if mask is None else "all (masked)",
            "period_or_history": f"t={t}",
            "mean": mu,
            "std": sd,
        })

    # bids
    for t in (1, 2, 3):
        b = np.asarray(result[f"b{t}_sample"])
        if mask is not None:
            b = b[mask]
        mu, sd = stats(b)
        rows.append({
            "category": "bid",
            "detail": "all" if mask is None else "all (masked)",
            "period_or_history": f"t={t}",
            "mean": mu,
            "std": sd,
        })

    # winning bids
    for t in (1, 2, 3):
        b = np.asarray(result[f"b{t}_sample"])

        if b.ndim == 2:
            if mask is not None and mask.shape == b.shape:
                b_masked = np.where(mask, b, np.nan)

                # auctions where at least one player is in the subgroup
                has_group = np.any(mask, axis=1)

                # nanmax ignores empty cells
                b_win = np.nanmax(b_masked[has_group], axis=1)
            else:
                b_win = b.max(axis=1)
        else:
            b_win = b

        mu, sd = stats(b_win)
        rows.append({
            "category": "winning_bid",
            "detail": "per_auction" if mask is None else "per_auction (masked)",
            "period_or_history": f"t={t}",
            "mean": mu,
            "std": sd,
        })

    # benefits
    b1 = np.asarray(result["b1_sample"])
    b2 = np.asarray(result["b2_sample"])
    b3 = np.asarray(result["b3_sample"])

    if mask is None:
        ben1 = _period_benefit(b1, V1_sample)
        ben2 = _period_benefit(b2, V2_sample)
        ben3 = _period_benefit(b3, V3_sample)
    else:
        ben1 = _period_benefit_for_group(b1, V1_sample, mask, zero_if_not_in_group=True)
        ben2 = _period_benefit_for_group(b2, V2_sample, mask, zero_if_not_in_group=True)
        ben3 = _period_benefit_for_group(b3, V3_sample, mask, zero_if_not_in_group=True)

    for t, ben in zip((1, 2, 3), (ben1, ben2, ben3)):
        mu, sd = stats(ben)
        rows.append({
            "category": "benefit",
            "detail": "winner_surplus" if mask is None else "winner_surplus (masked)",
            "period_or_history": f"t={t}",
            "mean": mu,
            "std": sd,
        })

    # rival budgets
    for detail, key, label in [
        ("rival | lose",      "B2_rival_dist_lose",     "t=2, lose"),
        ("rival | win-lose",  "B3_rival_dist_winlose",  "t=3, win-lose"),
        ("rival | lose-win",  "B3_rival_dist_losewin",  "t=3, lose-win"),
        ("rival | lose-lose", "B3_rival_dist_loselose", "t=3, lose-lose"),
    ]:
        mu, sd = stats(np.asarray(result[key]))
        rows.append({
            "category": "rival_budget",
            "detail": detail if mask is None else detail + " (unmasked)",
            "period_or_history": label,
            "mean": mu,
            "std": sd,
        })

    df = pd.DataFrame(
        rows,
        columns=["category", "detail", "period_or_history", "mean", "std"],
    )

    return df

def _empirical_pmf(
    data: np.ndarray, 
    grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical pmf of data over a given grid.
    Returns (grid_points, probabilities).
    """

    data = np.asarray(data).ravel()
    grid = np.asarray(grid)

    if grid.ndim != 1:
        raise ValueError("grid must be 1D")

    if len(grid) > 1:
        step = np.min(np.diff(grid))
    else:
        step = 1.0

    # bin edges around grid points
    bin_edges = np.concatenate((
        [grid[0] - step / 2.0],
        (grid[:-1] + grid[1:]) / 2.0,
        [grid[-1] + step / 2.0]
    ))

    counts, _ = np.histogram(data, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        pmf = np.zeros_like(counts, dtype=float)
    else:
        pmf = counts / total

    return grid, pmf

def _plot_three_periods(
    ax: plt.Axes,
    grid: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    label1: str,
    label2: str,
    label3: str,
    title: str,
    xlabel: str = "Value",
):
    """
    plot 3 distributions (period 1–3) on one axes.
    """
    g, p1 = _empirical_pmf(x1, grid)
    _, p2 = _empirical_pmf(x2, grid)
    _, p3 = _empirical_pmf(x3, grid)

    ax.plot(g, p1, marker=None, linestyle="-",  label=label1)
    ax.plot(g, p2, marker=None, linestyle="--", label=label2)
    ax.plot(g, p3, marker=None, linestyle=":",  label=label3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def _period_benefit(
    bids: np.ndarray,       # (N_A, N_P)
    values: np.ndarray      # (N_A, N_P)
) -> np.ndarray:            # (N_A,)
    
    """
    For each auction: winner's value - second highest bid.
    """

    bids = np.asarray(bids)
    values = np.asarray(values)

    if bids.shape != values.shape:
        raise ValueError("bids and values must have the same shape")
    
    if bids.ndim != 2:
        raise ValueError("Expected bids and values as 2D (auctions x players)")

    n_a, n_p = bids.shape
    if n_p < 2:
        raise ValueError("Need at least 2 players to define second-highest bid")

    order = np.argsort(bids, axis=1)
    winner_idx = order[:, -1]      # index of highest bid in each auction
    second_idx = order[:, -2]      # index of second-highest bid

    second_highest = bids[np.arange(n_a), second_idx]
    winner_values = values[np.arange(n_a), winner_idx]

    benefit = winner_values - second_highest
    
    return benefit

def _empirical_density(
    data: np.ndarray, 
    bins: int = 50
):
    """
    Simple empirical density from 1D data using a histogram.
    """

    data = np.asarray(data).ravel()

    if data.size == 0:
        return np.array([]), np.array([])

    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:]) # we want centers, not edges

    return centers, counts

def plot_all_distributions(
    result: dict,
    B_grid: np.ndarray, # (N_B, )
    mode: str,
    b_grid: np.ndarray | None = None, # in principle bid grid should equal budget grid
    rival_B_grid: np.ndarray | None = None,
):
    """
    Produce four figures:
      1) Budget distributions in periods 1–3 (all budgets)
      2) Bid distributions in periods 1–3 (all bids)
      3) Rival budget distributions (4 histories)
      4) Winning-bid distributions in periods 1–3 (per auction)
    """

    B_grid = np.asarray(B_grid)

    if b_grid is None:
        b_grid = B_grid
    else:
        b_grid = np.asarray(b_grid)

    if rival_B_grid is None:
        rival_B_grid = B_grid
    else:
        rival_B_grid = np.asarray(rival_B_grid)

    # budgets
    fig_bud, ax_bud = plt.subplots()
    _plot_three_periods(
        ax_bud,
        B_grid,
        result["B1_sample"],
        result["B2_sample"],
        result["B3_sample"],
        label1="Budget, period 1",
        label2="Budget, period 2",
        label3="Budget, period 3",
        title=f"Budget distributions over time. {mode}.",
        xlabel="Budget"
    )

    # bids
    fig_bid, ax_bid = plt.subplots()
    _plot_three_periods(
        ax_bid,
        b_grid,
        result["b1_sample"],
        result["b2_sample"],
        result["b3_sample"],
        label1="Bid, period 1",
        label2="Bid, period 2",
        label3="Bid, period 3",
        title=f"Bid distributions over time. {mode}.",
        xlabel="Bid"
    )

    # rival budgets
    fig_riv, ax_riv = plt.subplots()

    # all period-1 budgets
    g_base, p_B1_all = _empirical_pmf(result["B1_sample"], rival_B_grid)

    # rival distributions conditional on history
    g_r, p_B2_lose = _empirical_pmf(result["B2_rival_dist_lose"],      rival_B_grid)
    _,   p_B3_wl   = _empirical_pmf(result["B3_rival_dist_winlose"],   rival_B_grid)
    _,   p_B3_lw   = _empirical_pmf(result["B3_rival_dist_losewin"],   rival_B_grid)
    _,   p_B3_ll   = _empirical_pmf(result["B3_rival_dist_loselose"],  rival_B_grid)

    # Baseline B1 curve (no markers)
    ax_riv.plot(
        g_base, p_B1_all,
        marker=None,
        linestyle="-",
        linewidth=2.0,
        label="Budget, t=1",
    )

    # rival curves
    ax_riv.plot(g_r, p_B2_lose, marker=None, linestyle="--", label="Rival, t=2 | lose")
    ax_riv.plot(g_r, p_B3_wl,   marker=None, linestyle=":",  label="Rival, t=3 | win-lose")
    ax_riv.plot(g_r, p_B3_lw,   marker=None, linestyle="-.", label="Rival, t=3 | lose-win")
    ax_riv.plot(g_r, p_B3_ll,   marker=None, linestyle=(0, (3, 1, 1, 1)), label="Rival, t=3 | lose-lose")

    ax_riv.set_xlabel("Rival budget")
    ax_riv.set_ylabel("Probability")
    ax_riv.set_title(f"Rival budget distributions vs. initial budgets. {mode}.")
    ax_riv.legend()
    ax_riv.grid(True, alpha=0.3)

    # winning bids
    b1 = np.asarray(result["b1_sample"])
    b2 = np.asarray(result["b2_sample"])
    b3 = np.asarray(result["b3_sample"])

    b1_win = b1.max(axis=1) if b1.ndim == 2 else b1
    b2_win = b2.max(axis=1) if b2.ndim == 2 else b2
    b3_win = b3.max(axis=1) if b3.ndim == 2 else b3

    fig_win, ax_win = plt.subplots()
    _plot_three_periods(
        ax_win,
        b_grid,
        b1_win,
        b2_win,
        b3_win,
        label1="Winning bid, period 1",
        label2="Winning bid, period 2",
        label3="Winning bid, period 3",
        title=f"Winning bid distributions over time. {mode}.",
        xlabel="Winning bid"
    )

    return {
        "budgets": ax_bud,
        "bids": ax_bid,
        "rival_budgets": ax_riv,
        "winning_bids": ax_win,
    }

def plot_benefit_distributions(
    result: dict,
    V1_sample: np.ndarray, # (N_A, N_P)
    V2_sample: np.ndarray, # (N_A, N_P)
    V3_sample: np.ndarray, # (N_A, N_P)
    bins: int = 50,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot the distribution of winner benefits in each period.
    Benefit = winner's value - second-highest bid.
    """

    b1 = result["b1_sample"]
    b2 = result["b2_sample"]
    b3 = result["b3_sample"]

    ben1 = _period_benefit(b1, V1_sample)
    ben2 = _period_benefit(b2, V2_sample)
    ben3 = _period_benefit(b3, V3_sample)

    x1, d1 = _empirical_density(ben1, bins=bins)
    x2, d2 = _empirical_density(ben2, bins=bins)
    x3, d3 = _empirical_density(ben3, bins=bins)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x1, d1, marker=None, linestyle="-",  label="Benefit, period 1")
    ax.plot(x2, d2, marker=None, linestyle="--", label="Benefit, period 2")
    ax.plot(x3, d3, marker=None, linestyle=":",  label="Benefit, period 3")

    ax.set_xlabel("Winner benefit (winner value - second highest bid)")
    ax.set_ylabel("Density")
    ax.set_title(title or "Distribution of winner benefits over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def _plot_by_param_groups(
    ax: plt.Axes,
    grid: np.ndarray,
    data: np.ndarray, # (N_A, N_P)
    param: np.ndarray,# (N_A, N_P)
    param_groups: list[tuple[float, float]],
    group_labels: list[str],
    x_label: str,
    param_name: str,
    title: str | None = None,
) -> plt.Axes:
    """
    Generic helper: plot distributions of 'data' for several param-groups
    (e.g. beta or delta intervals) on the same axes.
    """

    if len(param_groups) != len(group_labels):
        raise ValueError("param_groups and group_labels must have the same length")

    grid = np.asarray(grid)
    data = np.asarray(data)
    param = np.asarray(param)

    if data.shape != param.shape:
        raise ValueError("data and param must have the same shape (N_AUCTIONS, N_PLAYERS)")
    if data.ndim != 2:
        raise ValueError("data must be 2D (auctions x players)")

    markers = ["o", "s", "^", "d", "v", ">"]
    linestyles = ["-", "--", ":", "-.", "-", "--"]

    for i, ((low, high), label) in enumerate(zip(param_groups, group_labels)):
        mask = (param >= low) & (param < high)
        data_group = data[mask]

        g, p = _empirical_pmf(data_group, grid)

        ax.plot(
            g,
            p,
            marker=None, #markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability")
    if title is not None:
        ax.set_title(title)
    if param_name:
        ax.legend(title=param_name)
    else:
        ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def plot_bids_by_param_groups(
    result: dict,
    period: int,
    bid_grid: np.ndarray,
    param: np.ndarray,
    param_groups: list[tuple[float, float]],
    group_labels: list[str],
    param_name: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:

    """
    Plot bid distributions in a given period for several param-groups
    (e.g. low/high beta or low/high delta) on the same figure.
    """

    b = np.asarray(result[f"b{period}_sample"])

    if ax is None:
        fig, ax = plt.subplots()

    return _plot_by_param_groups(
        ax=ax,
        grid=np.asarray(bid_grid),
        data=b,
        param=np.asarray(param),
        param_groups=param_groups,
        group_labels=group_labels,
        x_label="Bid",
        param_name=param_name,
        title=title or f"Bid distributions in period {period} by {param_name}-group",
    )

def plot_budgets_by_param_groups(
    result: dict,
    period: int,
    budget_grid: np.ndarray,
    param: np.ndarray,
    param_groups: list[tuple[float, float]],
    group_labels: list[str],
    param_name: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot budget distributions in a given period for several param-groups
    (e.g. low/high beta or low/high delta) on the same figure.
    """
    B = np.asarray(result[f"B{period}_sample"])

    if ax is None:
        fig, ax = plt.subplots()

    return _plot_by_param_groups(
        ax=ax,
        grid=np.asarray(budget_grid),
        data=B,
        param=np.asarray(param),
        param_groups=param_groups,
        group_labels=group_labels,
        x_label="Budget",
        param_name=param_name,
        title=title or f"Budget distributions in period {period} by {param_name}-group",
    )

def plot_solution_comparisons(
    results: dict,
    budget_grid: np.ndarray,
    bid_grid: np.ndarray | None = None,
    solution_labels: list[str] | None = None,
    mask: np.ndarray | None = None,
):
    """
    Compare multiple solution types (sophisticated, commitment, naive)
    by plotting, for each period t=1,2,3:
      - one figure with bid distributions for all solutions
      - one figure with budget distributions for all solutions
    """
    budget_grid = np.asarray(budget_grid)
    if bid_grid is None:
        bid_grid = budget_grid
    else:
        bid_grid = np.asarray(bid_grid)

    # make sure mask is a boolean array if given
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    # order of solutions
    solution_keys = list(results.keys())
    if solution_labels is None:
        solution_labels = solution_keys
    if len(solution_labels) != len(solution_keys):
        raise ValueError("solution_labels must have same length as results")

    # simple style cycles
    markers = ["o", "o", "o", "d", "v", ">"]
    linestyles = ["-", "--", ":", "-.", "-", "--"]

    axes_bids = {}
    axes_budgets = {}

    for t in (1, 2, 3):

        # bids
        fig_b, ax_b = plt.subplots()
        for i, (key, label) in enumerate(zip(solution_keys, solution_labels)):
            res = results[key]
            b_t = np.asarray(res[f"b{t}_sample"])
            if mask is not None:
                if mask.shape != b_t.shape:
                    raise ValueError("mask must have same shape as b_t_sample")
                b_t = b_t[mask]
            g, p = _empirical_pmf(b_t, bid_grid)

            ax_b.plot(
                g,
                p,
                marker=None,  # markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                label=label,
            )


        ax_b.set_xlabel(f"Bid, period {t}")
        ax_b.set_ylabel("Probability")
        ax_b.set_title(f"Bid distributions in period {t} by solution type")
        ax_b.legend()
        ax_b.grid(True, alpha=0.3)
        axes_bids[t] = ax_b

        # budgets
        fig_B, ax_B = plt.subplots()
        for i, (key, label) in enumerate(zip(solution_keys, solution_labels)):
            res = results[key]
            B_t = np.asarray(res[f"B{t}_sample"])
            if mask is not None:
                if mask.shape != B_t.shape:
                    raise ValueError("mask must have same shape as B_t_sample")
                B_t = B_t[mask]
            g, p = _empirical_pmf(B_t, budget_grid)

            ax_B.plot(
                g,
                p,
                marker=None,  # markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                label=label,
            )

        ax_B.set_xlabel(f"Budget, period {t}")
        ax_B.set_ylabel("Probability")
        ax_B.set_title(f"Budget distributions in period {t} by solution type")
        ax_B.legend()
        ax_B.grid(True, alpha=0.3)
        axes_budgets[t] = ax_B

    return {
        "bids": axes_bids,
        "budgets": axes_budgets,
    }

def save_figure(fig, subfolder: str, filename: str):
    """
    Save `fig` as `filename` inside FIG_DIR/subfolder, creating the folder if needed,
    and then close the figure.
    """
    folder = os.path.join(FIG_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)   # create Figures/subfolder if it doesn't exist
    full_path = os.path.join(folder, filename)
    fig.savefig(full_path)
    plt.close(fig)

def compute_benefit_stats_for_solution(
    result: dict,
    V1_sample: np.ndarray,
    V2_sample: np.ndarray,
    V3_sample: np.ndarray,
) -> dict:
    """
    Return mean benefits per period and overall for one solution.
    """
    ben1 = _period_benefit(result["b1_sample"], V1_sample)  # (N_A,)
    ben2 = _period_benefit(result["b2_sample"], V2_sample)
    ben3 = _period_benefit(result["b3_sample"], V3_sample)

    mean1 = float(np.mean(ben1))
    mean2 = float(np.mean(ben2))
    mean3 = float(np.mean(ben3))
    mean_all = float(np.mean(np.concatenate([ben1, ben2, ben3])))

    return {
        "mean_t1": mean1,
        "mean_t2": mean2,
        "mean_t3": mean3,
        "mean_all": mean_all,
    }

if __name__ == "__main__":

    if(N_P < 2):
        raise ValueError("Second price auctions must have at least two bidders")

    # create figures directory
    FIG_DIR = "figures"
    os.makedirs(FIG_DIR, exist_ok=True)

    # set rngs
    rng = default_rng(seed)
    rng_soph = default_rng(seed)
    rng_comm = default_rng(seed)

    # grids and draws
    beta_grid  = np.array(BETAS, dtype=float)
    delta_grid = np.array(DELTAS, dtype=float)
    B_grid, V_grid = make_grids(GRID_MIN, GRID_MAX, N_B, N_V)

    B1_draw    = draw_onto_grid(N_A, N_P, B_grid, rng)
    V1_draw    = draw_onto_grid(N_A, N_P, V_grid, rng)
    V2_draw    = draw_onto_grid(N_A, N_P, V_grid, rng)
    V3_draw    = draw_onto_grid(N_A, N_P, V_grid, rng)
    beta_draw  = draw_onto_grid(N_A, N_P, BETAS,  rng)
    delta_draw = draw_onto_grid(N_A, N_P, DELTAS, rng)

    B1_draw_big  = draw_onto_grid(N_A_big, N_P, B_grid, rng)
    V1_draw_big  = draw_onto_grid(N_A_big, N_P, V_grid, rng)
    V2_draw_big  = draw_onto_grid(N_A_big, N_P, V_grid, rng)
    V3_draw_big  = draw_onto_grid(N_A_big, N_P, V_grid, rng)
    beta_draw_big  = draw_onto_grid(N_A_big, N_P, BETAS, rng)
    delta_draw_big = draw_onto_grid(N_A_big, N_P, DELTAS, rng)

    _ , B1_sample_pmf = _empirical_pmf(B1_draw,B_grid)
    print(B1_sample_pmf)

    # solve sophisticated and commitment using iteration procedure
    result_sophisticated = three_period_solver(
        B_grid, V_grid, BETAS, DELTAS, B1_draw,
        V1_draw, V2_draw, V3_draw, beta_draw, delta_draw,
        rng_soph, KS_TOLERANCE, N_ITERATIONS, DAMPING,N_H2,N_H3, USE_PMF_RIVALS, "sophisticated"
    )

    result_commitment = three_period_solver(
        B_grid, V_grid, BETAS, DELTAS, B1_draw,
        V1_draw, V2_draw, V3_draw, beta_draw, delta_draw,
        rng_comm, KS_TOLERANCE, N_ITERATIONS, DAMPING,N_H2,N_H3, USE_PMF_RIVALS, "commitment"
    )

    """
    result_naive = three_period_solver(
        B_grid, V_grid, BETAS, DELTAS, B1_draw,
        V1_draw, V2_draw, V3_draw, beta_draw, delta_draw,
        rng, KS_TOLERANCE, N_ITERATIONS, DAMPING,N_H2,N_H3, USE_PMF_RIVALS, "naive"
    )
    """

    # build naive solution from the others
    result_naive = {}
    result_naive["Eflow2_interpolators"]      = result_commitment["Eflow2_interpolators"]
    result_naive["Eflow3from2_interpolators"] = result_commitment["Eflow3from2_interpolators"]
    result_naive["b2_strategy"] = result_sophisticated["b2_strategy"].copy()

    results = {
        "sophisticated": result_sophisticated,
        "commitment":    result_commitment,
        "naive":         result_naive,
    }



    # final big run
    final_results = {}
    
    rng_finals = {}
    rng_finals["sophisticated"] = default_rng(seed)
    rng_finals["commitment"] = default_rng(seed)
    rng_finals["naive"] = default_rng(seed)

    rng_common = default_rng(seed)
    rival_draws_big = rng_common.random(size=(N_A_big, N_R))

    for mode_name, res in results.items():
        print(f"Running large-sample simulation for: {mode_name}")

        b1_strategy_sample_big = b1_star_auctions(
            B_grid, V_grid,
            B1_draw_big, V1_draw_big, V2_draw_big, V3_draw_big,
            beta_draw_big, delta_draw_big,
            res["Eflow2_interpolators"],
            res["Eflow3from2_interpolators"]
        )

        (
            b1_sample_big,
            b2_sample_big,
            b3_sample_big,
            B2_sample_big,
            B3_sample_big,
            B2_rival_lose_big,
            B3_rival_winlose_big,
            B3_rival_losewin_big,
            B3_rival_loselose_big,
            pmf_lose_big,
            pmf_winlose_big,
            pmf_losewin_big,
            pmf_loselose_big,
        ) = forward_simulation(
            B_grid, V_grid, beta_grid, delta_grid,
            B1_draw_big, V1_draw_big, V2_draw_big, V3_draw_big,
            beta_draw_big, delta_draw_big,
            b1_strategy_sample_big,
            res["b2_strategy"],   # converged b2 strategy
            rival_draws_big,
            rng_finals[mode_name],
        )

        final_results[mode_name] = {
            "b2_strategy": res["b2_strategy"],
            "b1_sample": b1_sample_big,
            "b2_sample": b2_sample_big,
            "b3_sample": b3_sample_big,
            "B1_sample": B1_draw_big,
            "B2_sample": B2_sample_big,
            "B3_sample": B3_sample_big,
            "B2_rival_dist_lose": B2_rival_lose_big,
            "B3_rival_dist_winlose": B3_rival_winlose_big,
            "B3_rival_dist_losewin": B3_rival_losewin_big,
            "B3_rival_dist_loselose": B3_rival_loselose_big,
            "pmf_lose": pmf_lose_big,
            "pmf_winlose": pmf_winlose_big,
            "pmf_losewin": pmf_losewin_big,
            "pmf_loselose": pmf_loselose_big,
            "mode": mode_name,
            "N_A": N_A_big,
        }

        if GET_FULL_B1_STRATEGY:
            t_strat_start = perf_counter()
            print(f"  Started b1 strategy for {mode_name}")
            b1_strategy = b1_star_grid(
                B_grid, V_grid, beta_grid, delta_grid,
                res["Eflow2_interpolators"],
                res["Eflow3from2_interpolators"]
            )
            t_strat_finish = perf_counter()
            final_results[mode_name]["b1_strategy"] = b1_strategy
            print(f"  Finished b1 strategy for {mode_name}:",
                  t_strat_finish - t_strat_start, "seconds")

    """
    # ----------------------------------------------------
    # Numeric summaries on the *grid* (b1, b2 strategies)
    # focusing on low-β index
    # ----------------------------------------------------

    # Identify the low-β index in your grid, e.g. BETAS = [0.5, 1.0]
    BETAS = np.asarray(BETAS, dtype=float)
    low_beta_idx = int(np.argmin(BETAS))    # smallest β
    delta_idx = 0                           # if you only have one δ

    print("\n")

    # b1 strategies, if computed
    if "b1_strategy" in final_results["sophisticated"]:
        b1_soph = final_results["sophisticated"]["b1_strategy"]
        b1_comm = final_results["commitment"]["b1_strategy"]
        b1_naive = final_results["naive"]["b1_strategy"]

        # slice out the low-β, first-δ block
        b1_soph_low = b1_soph[..., low_beta_idx, delta_idx]
        b1_comm_low = b1_comm[..., low_beta_idx, delta_idx]
        b1_naive_low = b1_naive[..., low_beta_idx, delta_idx]

        mean_b1_soph_low = np.mean(b1_soph_low)
        mean_b1_comm_low = np.mean(b1_comm_low)
        mean_b1_naive_low = np.mean(b1_naive_low)

        print("Mean b1 on grid (low β):")
        print("  sophisticated:", mean_b1_soph_low)
        print("  commitment   :", mean_b1_comm_low)
        print("  naive        :", mean_b1_naive_low)

        max_diff_comm_naive = np.max(np.abs(b1_comm_low - b1_naive_low))
        print("Max |b1_comm - b1_naive| on grid (low β):", max_diff_comm_naive)
        print("\n")

    # b2 strategies (always on grid)
    b2_soph = final_results["sophisticated"]["b2_strategy"]
    b2_comm = final_results["commitment"]["b2_strategy"]
    b2_naive = final_results["naive"]["b2_strategy"]

    b2_soph_low = b2_soph[..., low_beta_idx, delta_idx]
    b2_comm_low = b2_comm[..., low_beta_idx, delta_idx]
    b2_naive_low = b2_naive[..., low_beta_idx, delta_idx]

    mean_b2_soph_low = np.mean(b2_soph_low)
    mean_b2_comm_low = np.mean(b2_comm_low)
    mean_b2_naive_low = np.mean(b2_naive_low)

    print("Mean b2 on grid (low β):")
    print("  sophisticated:", mean_b2_soph_low)
    print("  commitment   :", mean_b2_comm_low)
    print("  naive        :", mean_b2_naive_low)

    max_diff_soph_naive = np.max(np.abs(b2_soph_low - b2_naive_low))
    print("Max |b2_soph - b2_naive| on grid (low β):", max_diff_soph_naive)

    # benefit stats
    benefit_stats = {}
    solution_titles = ["sophisticated","commitment","naive"]

    for key in solution_titles:
        result = final_results[key]
        stats_sol = compute_benefit_stats_for_solution(
            result,
            V1_draw_big,
            V2_draw_big,
            V3_draw_big,
        )
        benefit_stats[key] = stats_sol

        print(f"\n{key.capitalize()} solution:")
        print(f"  E[benefit | t=1]  = {stats_sol['mean_t1']:.4f}")
        print(f"  E[benefit | t=2]  = {stats_sol['mean_t2']:.4f}")
        print(f"  E[benefit | t=3]  = {stats_sol['mean_t3']:.4f}")
        print(f"  E[benefit | all]  = {stats_sol['mean_all']:.4f}")

    print("\nOverall mean benefit differences (all periods pooled):")

    def diff(a, b):
        return benefit_stats[a]["mean_all"] - benefit_stats[b]["mean_all"]

    print(f"  Sophisticated - Commitment = {diff('sophisticated','commitment'):.4f}")
    print(f"  Sophisticated - Naive      = {diff('sophisticated','naive'):.4f}")
    print(f"  Commitment   - Naive       = {diff('commitment','naive'):.4f}")
    """

    print("\n\n")

    # --- summary statistics ---
    for name, result in final_results.items():

        mask_beta_more_09 = (beta_draw_big >= 0.9)
        mask_beta_less_09 = (beta_draw_big < 0.9)

        mask_delta_more_09 = (delta_draw_big >= 0.9)
        mask_delta_less_09 = (delta_draw_big < 0.9)

        rng_for_summary_statistics = default_rng(seed)

        print(f"{name} SOLUTION SUMMARY STATISTICS\n")

        # --- win rates by beta group ---
        b1 = np.asarray(result["b1_sample"])
        b2 = np.asarray(result["b2_sample"])
        b3 = np.asarray(result["b3_sample"])

        # period 1
        winners1 = realized_winners(b1,rng_for_summary_statistics)
        win_rate_low1  = mask_beta_less_09[np.arange(len(winners1)), winners1].mean()
        win_rate_high1 = mask_beta_more_09[np.arange(len(winners1)), winners1].mean()
        win_rate_lowd1  = mask_delta_less_09[np.arange(len(winners1)), winners1].mean()
        win_rate_highd1 = mask_delta_more_09[np.arange(len(winners1)), winners1].mean()

        # period 2
        winners2 = realized_winners(b2,rng_for_summary_statistics)
        win_rate_low2  = mask_beta_less_09[np.arange(len(winners2)), winners2].mean()
        win_rate_high2 = mask_beta_more_09[np.arange(len(winners2)), winners2].mean()
        win_rate_lowd2  = mask_delta_less_09[np.arange(len(winners2)), winners2].mean()
        win_rate_highd2 = mask_delta_more_09[np.arange(len(winners2)), winners2].mean()

        # period 3
        winners3 = realized_winners(b3,rng_for_summary_statistics)
        win_rate_low3  = mask_beta_less_09[np.arange(len(winners3)), winners3].mean()
        win_rate_high3 = mask_beta_more_09[np.arange(len(winners3)), winners3].mean()
        win_rate_lowd3  = mask_delta_less_09[np.arange(len(winners3)), winners3].mean()
        win_rate_highd3 = mask_delta_more_09[np.arange(len(winners3)), winners3].mean()
        
        print("\nWin rates (beta groups):")
        print(f"  Low beta 1: {win_rate_low1:.4f}")
        print(f"  High beta 1: {win_rate_high1:.4f}\n")
        print(f"  Low beta 2: {win_rate_low2:.4f}")
        print(f"  High beta 2: {win_rate_high2:.4f}\n")
        print(f"  Low beta 3: {win_rate_low3:.4f}")
        print(f"  High beta 3: {win_rate_high3:.4f}\n")

        print("\nWin rates (delta groups):")
        print(f"  Low delta 1: {win_rate_lowd1:.4f}")
        print(f"  High delta 1: {win_rate_highd1:.4f}\n")
        print(f"  Low delta 2: {win_rate_lowd2:.4f}")
        print(f"  High delta 2: {win_rate_highd2:.4f}\n")
        print(f"  Low delta 3: {win_rate_lowd3:.4f}")
        print(f"  High delta 3: {win_rate_highd3:.4f}\n")

        # --- no mask ---
        summary = summarize_outcomes(
            result,
            V1_sample=V1_draw_big,
            V2_sample=V2_draw_big,
            V3_sample=V3_draw_big,
        )
        print(summary.to_string(index=False))

        # --- beta < 0.9 ---
        print("\nLow Beta (high bias)\n")
        summary_beta_less = summarize_outcomes(
            result,
            V1_draw_big,
            V2_draw_big,
            V3_draw_big,
            mask=mask_beta_less_09
        )
        print(summary_beta_less.to_string(index=False))

        # --- beta ≥ 0.9 ---
        print("\nHigh Beta (low bias)\n")
        summary_beta_more = summarize_outcomes(
            result,
            V1_draw_big,
            V2_draw_big,
            V3_draw_big,
            mask=mask_beta_more_09
        )
        print(summary_beta_more.to_string(index=False))

        # --- delta < 0.9 ---
        print("\nLow Delta (more impatient)\n")
        summary_delta_less = summarize_outcomes(
            result,
            V1_draw_big,
            V2_draw_big,
            V3_draw_big,
            mask=mask_delta_less_09
        )
        print(summary_delta_less.to_string(index=False))

        # --- delta ≥ 0.9 ---
        print("\nHigh Delta (more patient)\n")
        summary_delta_more = summarize_outcomes(
            result,
            V1_draw_big,
            V2_draw_big,
            V3_draw_big,
            mask=mask_delta_more_09
        )
        print(summary_delta_more.to_string(index=False))
      
        # --- LaTeX output and saving ---
        def df_to_latex_rows(df, float_fmt="%.3f"):
            """
            Convert DataFrame to LaTeX *rows only* (no begin/end, no top/bottom rules).
            """
            latex_full = df.to_latex(index=False, header=False, float_format=float_fmt)
            lines = latex_full.strip().split("\n")

            data_lines = []
            for line in lines:
                if line.startswith(r"\begin{tabular"):
                    continue
                if line.startswith(r"\end{tabular"):
                    continue
                if line.startswith(r"\toprule"):
                    continue
                if line.startswith(r"\bottomrule"):
                    continue
                data_lines.append(line)

            return "\n".join(data_lines)

        latex_summary         = df_to_latex_rows(summary)
        latex_beta_less       = df_to_latex_rows(summary_beta_less)
        latex_beta_more       = df_to_latex_rows(summary_beta_more)
        latex_delta_less = df_to_latex_rows(summary_delta_less)
        latex_delta_more = df_to_latex_rows(summary_delta_more)

        print("\n\nLaTeX summary table:\n")
        print(latex_summary)
        print("\nLaTeX summary (beta less):\n")
        print(latex_beta_less)
        print("\nLaTeX summary (beta more):\n")
        print(latex_beta_more)
        print("\nLaTeX summary (delta less):\n")
        print(latex_delta_less)
        print("\nLaTeX summary (delta more):\n")
        print(latex_delta_more)
        print("\n\n")

        with open(f"summary_{name}.tex", "w") as f:
            f.write(latex_summary)

        with open(f"summary_beta_less_{name}.tex", "w") as f:
            f.write(latex_beta_less)

        with open(f"summary_beta_more_{name}.tex", "w") as f:
            f.write(latex_beta_more)

        with open(f"summary_delta_less_{name}.tex", "w") as f:
            f.write(latex_delta_less)

        with open(f"summary_delta_more_{name}.tex", "w") as f:
            f.write(latex_delta_more)

    # --- build graphs ---

    solution_order = ["sophisticated", "commitment", "naive"]
    solution_titles = {
        "sophisticated": "Sophisticated",
        "commitment":    "Commitment",
        "naive":         "Naive",
    }

    low_beta_mask_big = (beta_draw_big < 1.0)  # works for betas like [0.5, 1.0]

    axes_compare = plot_solution_comparisons(
        results=final_results,
        budget_grid=B_grid,
        bid_grid=B_grid,
        solution_labels=[solution_titles[k] for k in solution_order],
        mask=low_beta_mask_big,
    )

    for t, ax in axes_compare["bids"].items():
        fig = ax.figure
        save_figure(fig, subfolder="comparisons",
                    filename=f"bids_period{t}_comparison.pdf")

    for t, ax in axes_compare["budgets"].items():
        fig = ax.figure
        save_figure(fig, subfolder="comparisons",
                    filename=f"budgets_period{t}_comparison.pdf")

    for key in solution_order:
        result = final_results[key]
        mode_title = f"{solution_titles[key]} solution"

        axes = plot_all_distributions(
            result=result,
            B_grid=B_grid,
            mode=mode_title,
        )

        # Keys: "budgets", "bids", "rival_budgets", "winning_bids"
        for ax_name, ax in axes.items():
            fig = ax.figure
            save_figure(fig, subfolder="all_distributions",
                        filename=f"{ax_name}_{key}.pdf")

    beta_groups = [(0.0, 0.9), (0.9, 1.01)]
    beta_labels = ["low β", "high β"]

    for sol_key in solution_order:
        result = final_results[sol_key]
        sol_title = solution_titles[sol_key]

        for period in [1, 2, 3]:

            # bids
            ax_bids_beta = plot_bids_by_param_groups(
                result=result,
                period=period,
                bid_grid=B_grid,
                param=beta_draw_big,
                param_groups=beta_groups,
                group_labels=beta_labels,
                param_name=None,
                title=f"Period {period} bids by β group. {sol_title} solution.",
            )
            fig = ax_bids_beta.figure
            save_figure(fig, subfolder="beta_groups",
                        filename=f"bids_period{period}_beta_groups_{sol_key}.pdf")

            # budgets
            ax_budgets_beta = plot_budgets_by_param_groups(
                result=result,
                period=period,
                budget_grid=B_grid,
                param=beta_draw_big,
                param_groups=beta_groups,
                group_labels=beta_labels,
                param_name=None,
                title=f"Period {period} budgets by β group. {sol_title} solution.",
            )
            fig = ax_budgets_beta.figure
            save_figure(fig, subfolder="beta_groups",
                        filename=f"budgets_period{period}_beta_groups_{sol_key}.pdf")

    delta_groups = [(0.0, 0.9), (0.9, 1.01)]
    delta_labels = ["low δ", "high δ"]

    for sol_key in solution_order:
        result = final_results[sol_key]
        sol_title = solution_titles[sol_key]

        for period in [1, 2, 3]:

            # bids
            ax_bids_delta = plot_bids_by_param_groups(
                result=result,
                period=period,
                bid_grid=B_grid,
                param=delta_draw_big,
                param_groups=delta_groups,
                group_labels=delta_labels,
                param_name=None,
                title=f"Period {period} bids by δ group. {sol_title} solution.",
            )
            fig = ax_bids_delta.figure
            save_figure(fig, subfolder="delta_groups",
                        filename=f"bids_period{period}_delta_groups_{sol_key}.pdf")

            # budgets
            ax_budgets_delta = plot_budgets_by_param_groups(
                result=result,
                period=period,
                budget_grid=B_grid,
                param=delta_draw_big,
                param_groups=delta_groups,
                group_labels=delta_labels,
                param_name=None,
                title=f"Period {period} budgets by δ group. {sol_title} solution.",
            )
            fig = ax_budgets_delta.figure
            save_figure(fig, subfolder="delta_groups",
                        filename=f"budgets_period{period}_delta_groups_{sol_key}.pdf")

    for key in solution_order:
        result = final_results[key]
        name = solution_titles[key]

        ax_benefit = plot_benefit_distributions(
            result,
            V1_sample=V1_draw_big,
            V2_sample=V2_draw_big,
            V3_sample=V3_draw_big,
            bins=N_B,
            title=f"Distribution of winner benefits over time. {name} solution.",
        )
        fig = ax_benefit.figure
        save_figure(fig, subfolder="benefits",
                    filename=f"benefit_distributions_{key}.pdf")

    plt.show()

