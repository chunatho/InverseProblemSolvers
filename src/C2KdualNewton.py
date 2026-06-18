import time
import matplotlib.pyplot as plt
from numba import njit
from numpy import shape, reshape, array, diag, flip, copy
from numpy import linspace, zeros, zeros_like, ones
from numpy import sqrt, exp, abs, log10, log, argmax
from numpy import isfinite, nan
from numpy import any, logical_and, amax, amin
from numpy import sum, average, var
from numpy.linalg import inv
# Chi-sq kink algorithm
from scipy.optimize import least_squares
from DualPerspective import DPModel, solve, regularize

def C2KdN_solve(A, b, C, mu, \
                alpha_min=1e0, alpha_max=1e8, Nalpha = 19, \
                C2K_min=2, C2K_max=2.5, NC2K=10, \
                atol=1e-8, rtol=1e-8 ):
    # 1) This code optimizes ||A@x -b||^2_C + \alpha S_sj for Nalpha points
    # in the interval [alpha_min, alpha_max] using dual Newton optimization algorithm.
    # "Dual formulation of the maximum entropy method applied to analytic continuation of quantum Monte Carlo data"
    # Chuna, Thomas and Barnfield, Nicholas and Dornheim, Tobias and Friedlander, Michael and Hoheisel, Tim
    # doi: 10.1088/1751-8121/adf924
    # 2) Then uses the chi2kink algorithm to select a single best alpha value
    # and returns the solution at that alpha value.
    # "Algorithms for optimized maximum entropy and diagnostic tools for analytic continuation"
    # Bergeron, Dominic and Tremblay, A.-M.S.
    # doi:10.1103/PhysRevE.94.023303
    # "ana\_cont: Python package for analytic continuation"
    # Kaufmann, Josef and Held, Karsten
    # doi:10.1016/j.cpc.2022.108519
    # Input:
    #   A: matrix to be inverted.
    #   b: data
    #   C: data's convariance matrix.
    #   mu: default model or prior.
    #   alpha_min: smallest alpha value to test
    #   alpha_max: largest alpha value to test
    #   Nalpha: number of points in the alpha grid
    #   C2K_min (recommended 2.0): smaller include larger regularization parameters
    #   C2K_max (recommended 2.5: larger includes smaller regularization parameters
    #   NC2K (recommended 10+): Number of regularization weights to sample between min and max
    #   rtol: the relative tolerance of the solution ||Ax-b||^2
    #   atol: absolute tolerance for minimum of ||Ax-b||^2
    #
    # Output:
    #    x_avg: solution estimate
    #    x_err: stdev of solution w.r.t. small changes in alpha
    #    x_arr: arr of solutions (one for each alpha)
    #    chi2_arr: array of chi2 values (one for each alpha value)
    #    alpha_best_vals: the regularization parameters selected by the chi2kink algorithm.

    # Check the dimension of inputs
    [Ntau, Nomega] = shape(A) # Ntau rows and Nomega cols
    b = reshape(b, (Ntau, 1)) # make sure that data b is the right shape
    mu = reshape(mu, (Nomega, 1)) # make sure that Bayesian prior is the right shape.
    Cinv = inv(C) # precompute the inverse of the covariance matrix for computational savings.

    # Instantiate dual problem to solve
    Cinv = inv(C)
    model = DPModel(A, b[:,0], q=mu[:,0], C=C, scale=1.0)#, c=one_vector,λ=None)  # Optionally: DPModel(A, b, q=None, C=None, c=None, λ=None)

    # step 1: compute chi are a collection of alpha values
    x = zeros((Nomega,1)); # this list will be converted into an array! x_arr = zeros((Nalpha, Nomega));
    alphas = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha)
    chi2_arr = zeros((Nalpha, 2 )); # store the cost function min (i.e. x_alpha) for each alpha;
    chi2_arr[:,0] = alphas
    for i in range( Nalpha ):
        regularize(model, alphas[i])  # set regularization of optimization problem
        x[:,0] = solve(model, rtol=rtol, atol=atol, t0=1.0) # solve for alphas in domain

        if ( np_all(x >= 0.0) and np_all( isfinite(x) ) ):
            #store results
            res = A@x - b
            chi2_arr[i,1] = (res.T@Cinv@res)[0,0]/Ntau # eqn 3.6
        else:
            if ( not np_all(isfinite(x)) ):
                print('Invalid Solution for alpha ', alphas[i], ' there are NaNs or infs')
            if ( not np_all(x >= 0.0) ):
                print('Invalid solution for alpha ', alphas[i], ' there are negatives')
            chi2_arr[i,1] = nan

    # step 2: Fit logistic growth function and then determine alphas near the lower kink
    params, error = fit_logistic( log10(chi2_arr[:, 0]), log10(chi2_arr[:, 1]), 0.1*abs(log10(chi2_arr[:, 1])) )
    [_, _, c, d] = params

    #x = log10( chi2_arr[:,0] )
    #y = log10( chi2_arr[:,1] )
    #dy = 0.1*abs(log10(chi2_arr[:,1]))
    #params, error = fit_logistic( x, y, dy )

    chi2kink_vals  = linspace(C2K_min, C2K_max, NC2K)
    alpha_best_vals = 10**(c - chi2kink_vals/d)

    # Check for violations
    below_min = alpha_best_vals < alpha_min
    above_max = alpha_best_vals > alpha_max

    # Optional: distinguish cases
    if any(below_min):
        print(f'{sum(below_min)} values below alpha_min={alpha_min}')
    elif any(above_max):
        print(f'{sum(above_max)} values above alpha_max={alpha_max}')
    else:
        print('All alpha_best_vals are within bounds.')


    # step 3: compute solution using average over best alphas
    x_arr = zeros((Nomega, NC2K))
    for i, alpha_best in enumerate(alpha_best_vals):
        regularize(model, alpha_best)  # set regularization of optimization problem
        x_arr[:,i] = solve(model, rtol=rtol, atol=atol)

    x = reshape(average(x_arr, axis=1), (Nomega,1))
    x_var = reshape(var(x_arr, axis=1), (Nomega,1))
    return x, x_var, x_arr, chi2_arr, alpha_best_vals

@njit(cache=True)
def np_all(x):
    # Numba compatible version of np.all(x, axis=0)
    out = True
    for bool in x.flatten():
        out = logical_and(out, bool)
    return out

@njit(cache=True)
def lower_bound(x, eps):
    # Numba compatible version of x[ x < eps] = eps.
    for i, val in enumerate(x.flatten()):
        if ( val < eps):
            x[i] = eps
    return x

# Fit logitic function
def fit_logistic(x,y,dy):
    initial_guess = [0.0, 2.8, 10.0, 0.3]  # Initial guess for (a, b, c, d)
    res = least_squares(residuals, x0=initial_guess, args=(x, y, dy), loss='soft_l1', f_scale=3.0, max_nfev=10000)
    return res.x, None

# define the chi-sq residuals
def residuals(params, x, y, dy):
    a, b, c, d = params
    return (logistic_function(x, a, b, c, d) - y)/dy

# Define the logistic function
@njit
def logistic_function(alpha, a, b, c, d):
    return a + b / (1 + exp(-d * (alpha - c)))
