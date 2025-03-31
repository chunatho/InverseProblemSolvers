from numba import njit
from numpy import shape, reshape, array, diag
from numpy import linspace, geomspace, zeros, zeros_like, ones, eye, copy
from numpy import sqrt, exp, abs, log10, log, argmax, fmin
from numpy import isnan, nan, nan_to_num, isfinite, inf, bool8, float64
from numpy import any, logical_and
from numpy import sum
from numpy.linalg import svd, eigh, inv, solve
#from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import curve_fit

def C2KBryan_solve(A, b, C, mu, \
                   alpha_min=1e-3, alpha_max=1e3, Nalpha = 19, \
                   chi2kink_parameter=2.0, cond_upperbound = 1e10, \
                   max_iter=1000, max_fail=2000, numerical_zero=1e-16):
    # This code optimizes ||A@x -b||^2_C + \alpha S_sj,
    #  for Nalpha points in the interval [alpha_min, alpha_max] using Bryan's algorithm.
    # Bryan, R.K. "Maximum entropy analysis of oversampled data problems"
    # Eur. Biophys. J. 1990, 18, 165-174, doi:10.1007/BF02427376.
    # Then this program uses the chi2kin algorithm to select a single best alpha value
    # and returns the solution at that alpha value.
    # Input:
    #  A: matrix to be inverted.
    #  b: data
    #  C: data's convariance matrix.
    #  mu: default model or prior.
    #  alpha_min: smallest alpha value to test
    #  alpha_max: largest alpha value to test
    #  Nalpha: number of points in the alpha grid
    #  posterior_cutoff: Within the alpha domain [alpha_min, alpha_max] a given alpha is acceptable if
    #                    P(alpha)/max{P(\alpha)} > cut_off
    #  cond_upperbound: Cut off for maximum condition number of discretized kernel.
    #  max_iter: maximum number of Levenberg Marquart steps to take
    #  max_fail: maximum number of # of adjustments made to the LM parameter in an iteration
    #  numerical_zero: the logarithm will tend to produce wild results when x is very small.
    #                  so we introduce a cuoff make it behave better.
    # Output:
    #    x_avg: MEM estimate
    #    xsq_avg - x_avg*x_avg: the variance of MEM estimate
    #    x_arr: arr of solution (one for each alpha)
    #    P_arr: array of unnormalized Bayesian Posterior weighting function
    #    accepted_arr: array with 0s and 1s indicating whether a particular alpha was included in the averaging.

    # Check the dimension of inputs
    [Ntau, wcut_index] = shape(A) # Ntau rows and (wcut_index < Nomega) cols
    b = reshape(b, (Ntau, 1)) # make sure that data b is the right shape
    mu = reshape(mu, (wcut_index, 1)) # make sure that Bayesian prior is the right shape.
    Cinv = inv(C) # precompute the inverse of the covariance matrix for computational savings.

    x_arr = zeros((Nalpha, wcut_index)); # store the cost function min (i.e. x_alpha) for each alpha;
    chi2_arr = zeros((Nalpha, )); # store the cost function min (i.e. x_alpha) for each alpha;
    alphas = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha)
    for i in range( Nalpha ):
        x = Bryans_alg(A, b, C, mu, alphas[i], cond_upperbound, max_iter, max_fail)
        if ( np_all(x >= 0.0) and np_all( isfinite(x) ) ):
            #store results
            chi2_arr[i] = ( ( A@x - b ).T @ Cinv @ ( A@x - b ) )[0,0] # eqn 3.6
            x_arr[i] = x.flatten()
        else:
            if ( not np_all(isfinite(x)) ):
                print('Invalid Solution for alpha ', alphas[i], ' there are NaNs or infs')
            if ( not np_all(x >= 0.0) ):
                print('Invalid solution for alpha ', alphas[i], ' there are negatives')
            chi2_arr[i] = inf
    print('chi2_arr', log10(chi2_arr))
    # step 2: compute solution using weighted average over alpha
    params, error = fit_logistic( log10(alphas), log10(chi2_arr) )
    [_, _, c, d] = params
    alpha_best = 10**(c - chi2kink_parameter/d)
    print('params', params, 'alpha best', alpha_best)
    x = Bryans_alg(A, b, C, mu, alpha_best, cond_upperbound, max_iter, max_fail)
    return x, x_arr, chi2_arr, params, error

# Define the logistic function
@njit
def logistic_function(alpha, a, b, c, d):
    return a + b / (1 + exp(-d * (alpha - c)))

# Fit the data using curve_fit
def fit_logistic(alpha, y):
    initial_guess = [1.0, 1.0, 0.0, 1.0]  # Initial guess for (a, b, c, d)
    params, pcov = curve_fit(logistic_function, alpha, y, p0=initial_guess, maxfev=10000)
    return params, sqrt(diag(pcov))

@njit(cache=True)
def np_all(x):
    # Numba compatible version of np.all(x, axis=0)
    out = True
    for bool in x.flatten():
        out = logical_and(out, bool)
    return out

@njit
def Bryans_alg(A, b, C, mu, alpha: float64, cond_upperbound,  max_iter, max_fail):
    # Bryan's Algorithm published - https://link.springer.com/article/10.1007/BF02427376
    # This code follows the convention of Asakawa et al. https://arxiv.org/abs/hep-lat/0011040
    # Input:
    #   A: matrix to be inverted.
    #   b: data
    #   C: data's convariance matrix.
    #   mu: default model or prior.
    #   alpha: weight of the entropic prior.
    #   cond_upperbound: Cut off for maximum condition number of discretized kernel.
    #   max_iter: maximum number of Levenberg Marquart steps to take
    #   max_fail: maximum number of # of adjustments made to the LM parameter in an iteration
    # Output:
    #     x: solution

    [Ntau,_] = shape(A)
    # Decompose the convariance matrix
    eigvals, R = eigh(C) # C = R Lambda R.T (note Cinv = R lambda^{-1} R.T )
    eigvals_inv_matrix = diag(1./eigvals)

    # Rotate into diagonal covariance space (eqn 3.39)
    Arot = copy(R.T @ A)
    brot = copy(R.T @ b)
    #brot = copy(R.T @ reshape(b, (Ntau,1)) )

    # let svd produce A = V@diag(S)@Uh, thus we have to transpose Vh to get V
    # A=V@S@U.T disagrees with the typical A=U@S@V.T, but it is
    # how Asakawa writes it (see the text below eqn 3.30).
    V,S,Uh = svd(Arot); U=Uh.T
    # pre-condition the inversion to a set condition number
    if (cond_upperbound < 0.0):
        r=Ntau;
    else:
        #print('Pre-conditioning A in the primal alg so that kappa<',cond_upperbound)
        for i in range(Ntau):
            if S[0]/S[i] < cond_upperbound :
                r=i+1
            else:
                break
    V=V[:,0:r]; U=U[:,0:r]; S = diag(S[0:r])
    #DDL = eigvals_inv_matrix
    # Precompute the M for computational savings (eqn 3.37)
    M = S @ V.T @ (eigvals_inv_matrix) @ V @ S

    #y = ones((r,1), dtype='float64')
    y = zeros((r,1), dtype='float64')
    err_list=[ chi_sq(Arot, brot, eigvals_inv_matrix, mu, U, y) ];

    #initialize algorithm constants
    up=2.0; c0=0.1; i=0;
    while (i < max_iter):
        i+=1 # this index keeps track of the number of update iteration
        lam=1e-3; # this is the Levenberg-Marquardt (LM) parameter, it's like a learning rate.
        fail=0; # this index tracks the # of adjustments made to the LM parameter this iteration

        # ------------------------------------------------------- #
        # Compute RHS of eqn 3.36, where b has been relabeled `y' #
        # ------------------------------------------------------- #
        x = mu*exp(U @ y) # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy
        DL = eigvals_inv_matrix @ ( Arot @ x - brot ) # derivative of Likelihood (L) w.r.t. A@x
        g = S @ V.T @ DL # compute g from 3.34
        RHS = -alpha * y - g # compute the RHS of eqn 3.36

        # -------------------------------------------------------------- #
        # Try many different values for the Levenberg-Marquart parameter #
        # until an acceptable update is found of max_fail is reached.    #
        # -------------------------------------------------------------- #
        T = U.T @ diag(x.flatten()) @ U # pre-compute T from eqn 3.37
        while ( fail < max_fail ):
            LHS = (alpha + lam) * eye(r) + M @ T # compute the LHS of eqn 3.36
            delta = solve(LHS,RHS) # leave it up to numpy on how to solve it
            new_y = y + delta
            err = chi_sq(Arot, brot, eigvals_inv_matrix, mu, U, new_y)
            if ( isnan(err) or err_list[-1] < err): # only update if Chi_sq improves!
            #if ( (delta.T @ T @ delta)[0,0] > c0*sum(mu) ): # Asakawa update
                lam*=up;
                fail+=1;
            else: # successful update - reset Lev. Marq. parameter
                lam=1e-3
                break

        # check if fail limit has been reached
        if (fail == max_fail):
            print(' --- reached max fail! -- ')
            print('round', i, 'max iter', max_iter, 'abs(err_list[-2] - err) / err < 1e-5', abs(err_list[-2] - err) / err)
            break
        #update and check for convergence
        else:
            err_list.append(err);
            y=new_y;
            if ( abs(err_list[-2] - err) / err < 1e-5 ):
                break

    #print('round', i, 'max iter', max_iter, 'abs(err_list[-2] - err) / err < 1e-5', abs(err_list[-2] - err) / err)
    return mu*exp(U @ y); # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy


@njit
def chi_sq(A, b, eigvals_inv_matrix, mu, U, y):
    x = mu*exp(U @ y) # combining eqns (3.33) a = Uy and (3.29) x = m * exp(a)
    return sum((A@x-b).T @ eigvals_inv_matrix @ (A@x-b))


