import numpy as np
import sys
from numpy import random
from numba import njit
from numpy import shape, reshape, array, diag, tile, vstack
from numpy import any, all, round
from numpy import linspace, zeros, zeros_like, ones, eye, copy
from numpy import sqrt, exp, abs, log10, log, pi, nan, nan_to_num, inf
from numpy import sum, average
#from numpy.linalg import inv, solve, cond, norm, eigh
from numpy.linalg import inv # for Cinv in chi-square
# initialize kernel subfunction
from numpy import cumsum, amin, amax, diff, interp, clip
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, interp1d
#import matplotlib.pyplot as plt


# SACPT stands for stochastic autocorrelation with Parallel tempering (SACPT)
def Beach_solve(K, b, C, mu, omegas, taus, \
                ##### choose M walkers, P replicas, geometric factor R, initial configs C_p
                beta_min=1e-3, beta_max=1e3, Nwalkers=4, \
                max_iter=1000, min_stochastic_samples=2, tol=1e-1, \
                Ndeltas=102, Nupdates=50, step_size=0.1, \
                nearest_neighbors=5, seed=14729,
               ):
    # This code solves a matrix inversion using a parallel tempering algorithm
    #   ||A@x - b||^2_C,
    # it is implemented using Beach's seminal paper.
    # "Identifying the MEM as a special limit of SAC."
    # by K.S.D. Beach (2004).
    # DOI: 10.48550/arXiv.cond-mat/0403055
    #
    # Input:
    #   K: callable function that defines the kernel
    #   b: data
    #   C: data's convariance matrix.
    #   mu: default model or prior.
    #   omegas: grid of points where x(omega_j) is evaluated.
    #   taus: grid of points where b(tau_i) is evaluated.
    #   beta_min: smallest MH beta
    #   beta_max: largest MH beta
    #   step_size: asdf
    #   max_iter: maximum number of Levenberg Marquart steps to take
    #   num: number of kernels to use
    #   nearest_neighbors: maximum number of Levenberg Marquart steps to take
    #
    # Output:
    #     x_avg: MEM estimate
    #     xsq_avg - x_avg*x_avg: the variance of MEM estimate
    #     x_arr: arr of solution (one for each alpha)
    #     P_arr: array of unnormalized Bayesian Posterior weighting function
    #     accepted_arr: array with 0s and 1s indicating whether a particular alpha was included in the averaging.
    # There are many stochastic optimization methods, but this is our SOM.

    # Check the dimension of inputs
    Ntau, Nomega = len(taus), len(omegas) # Ntau rows and Nomega cols
    b = reshape(b, (Ntau, 1)) # make sure that data b is the right shape
    mu = reshape(mu, (Nomega, 1)) # make sure that Bayesian prior is the right shape.
    Cinv = inv(C) # precompute the inverse of the covariance matrix for computational savings.
    betas = 10**linspace(log10(beta_min), log10(beta_max), Nwalkers)
    x_arr = zeros((Nwalkers,Nomega))
    rng = random.default_rng(seed=seed)

    # create map phi(omega)=x and initalize walkers in x space.
    centers, coeffs, phi, phi_inv = initize_walker(omegas, mu[:,0], Ndeltas)
    delta_positions = tile(centers, (Nwalkers, 1))
    delta_weights = tile(coeffs, (Nwalkers, 1))
    Khat = K(taus, phi_inv(centers))
    chi2_arr = compute_chi2(Khat, reshape(coeffs,(-1,1)), b, Cinv)*ones((Nwalkers,))

    # declare matrices to accumulate the accepted values.
    avg_positions = zeros((Nwalkers, Ndeltas))
    avg_weights = zeros((Nwalkers, Ndeltas))
    num_accepted = zeros((Nwalkers,))
    for _ in range(max_iter):
        # Introduce stochastic noise to each walker
        for i in range(Nwalkers):
            Khat_test = K(taus, phi_inv(delta_positions[i]))
            # each walker make Nupdates MH attempts per swap attempt
            for j in range(Nupdates):
                # propose local move on walker[i] trial
                if rng.random() < 0.2: # 20% of the time change position
                    #delta_positions_in_omega = phi_inv(delta_positions[i])
                    new_positions = phi(propose_positions(phi_inv(copy(delta_positions[i])), rng, step_size, max_val=omegas[-1]))
                    Khat_test = K(taus, phi_inv(new_positions))
                    positions_update = True
                else: # 70% of the time change weights of kernel combination
                    new_weights = propose_weights_via_Nbody(delta_positions[i], copy(delta_weights[i]), rng, Nbody=3)
                    #new_weights = propose_weights(copy(delta_weights[i]), rng)
                    positions_update = False

                # Evaluate the chi2 of the proposed update
                new_chi2 = compute_chi2(Khat_test, delta_weights[i], b, Cinv)
                delta_chi2 = new_chi2 - chi2_arr[i]
                # Metropolis Hasting (MH) accept-reject
                if rng.uniform() < exp(-betas[i] * delta_chi2): # automatically true for delta_chi2 < 0
                    print('walker', i, 'accepted new_chi2 (old, new)', chi2_arr[i], new_chi2)
                    # accepted: update walker! C[p], h[p], H[p]
                    chi2_arr[i] = new_chi2
                    if positions_update :
                        delta_positions[i] = new_positions
                    else:
                        delta_weights[i] = new_weights

        # Parallel Tempering
        print('Parallel Temper: current chi2_arr', chi2_arr)
        for i in range(Nwalkers-1):
            rate = exp((betas[i] - betas[i+1])*(chi2_arr[i] - chi2_arr[i+1]))
            if rng.uniform() < rate:
                print('accepted a PT switch')
                # swap positions
                tmp = copy(delta_positions[i])
                delta_positions[i] = copy(delta_positions[i+1])
                delta_positions[i+1] = tmp
                # swap weights
                tmp = copy(delta_weights[i])
                delta_weights[i] = copy(delta_weights[i+1])
                delta_weights[i+1] = tmp
                # swap chi2
                tmp = copy(chi2_arr[i])
                chi2_arr[i] = copy(chi2_arr[i+1])
                chi2_arr[i+1] = tmp
                print('chi2_arr after PT switch', chi2_arr)

        # accumulate samples
        for i in range(Nwalkers-1):
            if (chi2_arr[i] < tol ):
                avg_positions[i] += delta_positions[i]
                avg_weights[i] += delta_weights[i]
                num_accepted[i] += 1
        print('num_accepted', num_accepted)

        # Check is each walker has colleccted the minium number of samples.
        if all(num_accepted > min_stochastic_samples):
            break

    for i in range(Nwalkers):
        avg_positions[i] /= num_accepted[i]
        avg_weights[i] /= num_accepted[i]
        # interpolate to Bayesian prior's omega grid, use phi_inv(x)=omega
        x_arr[i] = interp(omegas, phi_inv(avg_positions[i]), avg_weights[i])
        residual = K(taus,omegas)@reshape(x_arr[i], (-1,1)) - b
        chi2_arr[i] = (1./Ntau)*(residual.T@Cinv@residual)[0,0]

    x_avg = average(x_arr, axis=0)
    xsq_avg = average(x_arr**2, axis=0)
    return x_avg, xsq_avg - x_avg*x_avg, x_arr, chi2_arr


def initize_walker(w, D_w, Nkernels, nearest_neighbors=5):
    # Select distribution of kernel centers based on where D(w) has most mass.
    # Inputs:
    #     w: 1D array of input values.
    #     D_w: 1D array of function values corresponding to w.
    #     Nkernels (int): Number of points in the new redistributed array.
    #     nearest_neighbors (int): Number of points to average the defailt model.
    #                              The moves the grid selection toward uniform.
    #
    # Returns:
    #     new_w (np.ndarray): New w values redistributed based on the CDF.


    # step 1: Compute normalized discrete CDF
    #cumulative = cumsum(D_w)
    #phi_vals = cumulative / cumulative[-1]  # normalized to [0,1]
    # test identitiy map?!
    phi_vals = w # identity map

    # step 2: Create callable CDF phi, and inv-CDF inv_phi
    phi_spline = CubicSpline(w, phi_vals, bc_type='natural')
    phi = lambda x: phi_spline(x)
    phi_inv_spline = CubicSpline(phi_vals, w, bc_type='natural')
    phi_inv = lambda x: phi_inv_spline(x)

    # Step 3: Choose Dirac delta positions and weights based on CDF
    #centers = linspace(0, 1, Nkernels)
    centers = linspace(amax(phi_vals)/Nkernels, amax(phi_vals), Nkernels)
    coeffs = interp(phi_inv(centers), w, D_w)

    print('identitiy map test', w, phi(w), centers, phi(centers))
    return centers, coeffs, phi, phi_inv

def create_scaled_K_matrix(K, taus, centers, phi_inv):
    Ntau, Nx = len(taus), len(centers)
    Khat = zeros((Ntau, Nx))
    omega_from_x = phi_inv(centers) # convert centers to omegas
    for i, tau in enumerate(taus):
        Khat[i, :] = K(tau, phi_inv(centers))
    return Khat

def compute_chi2(Khat, x, b, Cinv):
    Ntau = len(b)
    b_x = Khat@x
    diff = b_x-b
    return (1./Ntau)*(diff.T@Cinv@diff)[0,0]

def propose_positions(centers, rng, step_size, max_val=1.0):
    # add Gaussian noise scaled by the difference between positions
    Nkernels = len(centers)
    # create noise
    stds = copy(centers)
    stds[1:] = diff(centers)
    gaussian_noise_vals = step_size*rng.normal(scale=stds)

    # apply noise as long as it does not cause kernels to change order.
    proposal_centers = copy(centers)
    for i in range(Nkernels):
        if (i == 0):
            lower = 0.0
            upper = centers[i+1]
        elif (i == Nkernels -1):
            lower = proposal_centers[i-1]
            upper = max_val
        else:
            lower = proposal_centers[i-1]
            upper = centers[i+1]
        test = centers[i] + gaussian_noise_vals[i]
        if ( test < upper and test > lower ):
            proposal_centers[i] = test
    return proposal_centers

def propose_weights(weights, rng):
    # randomly swap some weight between different dirac deltas
    i, j = rng.choice(len(weights), size=2, replace=False)
    max_move = min( 0.9*weights[i], 0.9*weights[j])
    delta = rng.normal(scale=0.5*max_move)
    delta = clip(delta, -max_move, max_move)
    weights[i] -= delta
    weights[j] += delta
    if(weights[i] < 0 or weights[j] < 0):
        print('ERROR: proposed redistribution makes weight negative')
        sys.exit(1)
    return weights


def propose_weights_via_Nbody(positions, weights, rng, Nbody=3):
    #print('start: sum of weights', sum(weights) )
    #print('start: average over omega', sum(weights*positions) )
    #print('start: variance over omega', sum(weights*positions**2) )

    # randomly swap some weight between different dirac deltas
    # --- randomly select subset of size Nmoments+1 ---
    Nmoments = Nbody-1
    idx = rng.choice(len(positions), size=Nbody, replace=False)

    # --- construct moment matrix ---
    # A_{k,λ} = a_λ^k for k=0..Nmoments-1
    A = vstack([positions[idx]**k for k in range(Nmoments)])

    # --- compute nullspace direction (SVD) ---
    _, _, vh = np.linalg.svd(A)
    Q = vh[-1]  # last singular vector spans nullspace
    Q /= np.linalg.norm(Q)

    # --- scale step so all r' remain positive ---
    eps_neg = np.min(np.where(Q < 0, positions[idx] / -Q, np.inf))
    eps_pos = np.min(np.where(Q > 0, positions[idx] / Q, np.inf))

    # pick random step within safe bounds
    eps = rng.uniform(0, 0.5*amin([eps_neg, eps_pos]))
    weights[idx] += eps * Q

    #print('end: sum of weights', sum(weights) )
    #print('end: average over omega', sum(weights*positions) )
    #print('end: variance over omega', sum(weights*positions**2) )

    return weights


#@njit
def Beach_propose_weights_via_Nbody(positions, weights, rng, Nbody=3):
    print('sum of weights start', sum(weights) )
    # randomly swap some weight between different dirac deltas
    idx = rng.choice(len(weights), size=Nbody, replace=False)

    # determine how much weight to swap
    Q = ones((Nbody,))
    Q[0] = -1
    prod = 1
    for i in range(1, Nbody):
        prod *= positions[idx[i]] - positions[idx[0]]
    for i in range(1, Nbody):
        Q[i] = prod
        for j in range(1,Nbody):
            if( j != i):
                Q[i] /= (positions[idx[j]] - positions[idx[i]])

    #set limits to prevent creating negavtive value.
    pos_flag = True
    neg_flag = True
    for i, val in enumerate(weights[idx]/Q):
        print('vals', val)
        if( val > 0 ):
            if(pos_flag):
                least_pos_val = val
                pos_flag=False
            else:
                least_pos_val = amin([least_pos_val, val])
        if( val < 0 ):
            if(neg_flag):
                least_neg_val = val
                neg_flag=False
            else:
                least_neg_val = amin([least_neg_val, val])

    s = rng.uniform(least_neg_val, least_pos_val)
    norm=0
    for i in range(Nbody):
        weights[idx[i]] -= s*Q[i]
        norm -= s*Q[i]
    print('norm after higher moment update', norm )
    print('sum of weights end', sum(weights) )
    return weights
