from numba import njit
from numpy import shape, reshape, array, diag
from numpy import linspace, zeros, ones, eye, copy
from numpy import sqrt, exp, abs, log10, log, argmax, fmin
from numpy import isnan, nan, isfinite, inf, float64
from numpy import logical_and
from numpy import sum, average, var
from numpy.linalg import svd, eigh, inv, solve
from scipy.linalg import cholesky, solve_triangular   # NEW: replaces eigh usage
from scipy.optimize import least_squares


def C2KBryan_solve(A, b, C, mu, \
                   alpha_min=1e-3, alpha_max=1e3, Nalpha=19, \
                   C2K_min=2, C2K_max=2.5, NC2K=10, cond_upperbound=1e14, \
                   max_iter=1000, max_fail=2000, numerical_zero=1e-16):
    # 1) This code optimizes ||A@x -b||^2_C + \alpha S_sj for Nalpha points
    # in the interval [alpha_min, alpha_max] using Bryan's optimization algorithm.
    # "Maximum entropy analysis of oversampled data problems"
    # Bryan, R.K.
    # doi:10.1007/BF02427376.
    #
    # 2) Then uses the chi2kink algorithm to select a single best alpha value
    # and returns the solution at that alpha value.
    # "Algorithms for optimized maximum entropy and diagnostic tools for analytic continuation"
    # Bergeron, Dominic and Tremblay, A.-M.S.
    # doi:10.1103/PhysRevE.94.023303
    # "ana\_cont: Python package for analytic continuation"
    # Kaufmann, Josef and Held, Karsten
    # doi:10.1016/j.cpc.2022.108519
    #
    # Input:
    #  A: matrix to be inverted.
    #  b: data
    #  C: data's convariance matrix.
    #  mu: default model or prior.
    #  alpha_min: smallest alpha value to test
    #  alpha_max: largest alpha value to test
    #  Nalpha: number of points in the alpha grid
    #  C2K_min=2: smaller include larger regularization parameters
    #  C2K_max=2.5: larger includes smaller regularization parameters
    #  NC2K=10: Number of regularization weights to sample between min and max
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

    # --- Cholesky whitening (replaces eigh + Cinv) ----------------------------
    # Decompose C = L @ L.T once, then whiten: Aw = L^{-1} A,  bw = L^{-1} b.
    # In the whitened system the covariance is the identity, so
    #   ||A x - b||^2_C  =  ||Aw x - bw||^2
    # and no eigvals_inv_matrix is needed anywhere downstream.
    # Doing this once here avoids repeating the factorisation inside every
    # Bryans_alg call (Nalpha + NC2K calls total).
    L  = cholesky(C, lower=True)                          # C = L L^T
    Aw = solve_triangular(L, A, lower=True)               # L^{-1} A
    bw = solve_triangular(L, b, lower=True)               # L^{-1} b
    # --------------------------------------------------------------------------

    x_arr    = zeros((Nalpha, wcut_index))
    chi2_arr = zeros((Nalpha,))
    alphas   = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha)

    for i in range(Nalpha):
        x = Bryans_alg(Aw, bw, mu, alphas[i], cond_upperbound, max_iter, max_fail)
        if ( np_all(x >= 0.0) and np_all(isfinite(x)) ):
            # chi^2 = ||Aw x - bw||^2  (identity covariance in whitened space)
            r = Aw @ x - bw
            print('r.T@r shape:', shape(r.T @ r))
            chi2_arr[i] = (r.T @ r)[0, 0]                 # simplified from original
        else:
            if not np_all(isfinite(x)):
                print('Invalid Solution for alpha ', alphas[i], ' there are NaNs or infs')
            if not np_all(x >= 0.0):
                print('Invalid solution for alpha ', alphas[i], ' there are negatives')
            chi2_arr[i] = nan

    # step 2: Fit logistic growth function and then determine alphas near the lower kink
    params, error = fit_logistic( log10(alphas), log10(chi2_arr), 0.1*abs(log10(chi2_arr)) )
    [_, _, c, d] = params

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


    # Step 3: Compute solutions near the lower kink and estimate error.
    x_arr = zeros((wcut_index, NC2K))
    for i, alpha_best_val in enumerate(alpha_best_vals):
        x_arr[:,i] = Bryans_alg(Aw, bw, mu, alpha_best_val, cond_upperbound, max_iter, max_fail).flatten()
    x     = reshape(average(x_arr, axis=1), (-1, 1))
    x_var = reshape(var(x_arr,     axis=1), (-1, 1))

    return x, x_var, x_arr, chi2_arr, alpha_best_vals


# Fit logistic function
def fit_logistic(x, y, dy):
    initial_guess = [0.0, 2.8, 10.0, 0.3]
    res = least_squares(residuals, x0=initial_guess, args=(x, y, dy), loss='soft_l1', f_scale=3.0, max_nfev=10000)
    return res.x, None

@njit
def residuals(params, x, y, dy):
    a, b, c, d = params
    return (logistic_function(x, a, b, c, d) - y) / dy

@njit
def logistic_function(alpha, a, b, c, d):
    return a + b / (1 + exp(-d * (alpha - c)))

@njit(cache=True)
def np_all(x):
    out = True
    for bool in x.flatten():
        out = logical_and(out, bool)
    return out


def Bryans_alg(Aw, bw, mu, alpha: float64, cond_upperbound, max_iter, max_fail):
    # Bryan's Algorithm — https://link.springer.com/article/10.1007/BF02427376
    # Convention follows Asakawa et al. https://arxiv.org/abs/hep-lat/0011040
    #
    # This version receives the *whitened* kernel Aw = L^{-1} A and right-hand
    # side bw = L^{-1} b, so the covariance in the whitened space is the
    # identity.  The eigvals_inv_matrix that appeared throughout the original
    # formulation therefore drops out entirely:
    #
    #   DL   = Aw @ x - bw           (was eigvals_inv_matrix @ (Arot@x - brot))
    #   M    = S @ S                  (was S @ V.T @ eigvals_inv_matrix @ V @ S;
    #                                  simplifies because V.T @ V = I_r after
    #                                  column truncation)
    #   chi_sq uses ||Aw x - bw||^2  (was (…)^T eigvals_inv_matrix (…))
    #
    # Input:
    #   Aw: whitened kernel  L^{-1} A  (Ntau x wcut_index)
    #   bw: whitened data    L^{-1} b  (Ntau x 1)
    #   mu: default model / prior      (wcut_index x 1)
    #   alpha: weight of the entropic prior
    #   cond_upperbound: cut-off for maximum condition number
    #   max_iter: maximum number of Levenberg–Marquardt steps
    #   max_fail: maximum number of LM-parameter adjustments per iteration
    # Output:
    #   x: solution  (wcut_index x 1)

    [Ntau, _] = shape(Aw)

    # SVD of the whitened kernel (replaces eigh + rotation + SVD of Arot)
    # Convention: Aw = V @ diag(S) @ Uh,  U = Uh.T  (Asakawa notation)
    V, S, Uh = svd(Aw); U = Uh.T

    # Pre-condition: keep only singular values whose ratio to S[0] is within
    # the allowed condition number bound.
    if cond_upperbound < 0.0:
        r = Ntau
    else:
        for i in range(Ntau):
            if S[0] / S[i] < cond_upperbound:
                r = i + 1
            else:
                break
    V = V[:, 0:r];  U = U[:, 0:r];  S = diag(S[0:r])

    # M from eqn 3.37, whitened form.
    # Original: M = S @ V.T @ eigvals_inv_matrix @ V @ S
    # After whitening eigvals_inv_matrix → I, and V.T @ V = I_r (orthonormal
    # columns after truncation), so M collapses to S @ S.
    M = S @ S                                              # diagonal r×r matrix

    y        = zeros((r, 1), dtype='float64')
    err_list = [chi_sq(Aw, bw, mu, U, y)]

    up = 2.0;  i = 0
    while i < max_iter:
        i += 1
        lam  = 1e-3
        fail = 0

        x  = mu * exp(U @ y)                              # eqns (3.29) & (3.33)
        DL = Aw @ x - bw                                  # whitened residual (no matrix multiply)
        g  = S @ V.T @ DL                                 # eqn 3.34
        RHS = -alpha * y - g                              # RHS of eqn 3.36

        T = U.T @ diag(x.flatten()) @ U                   # eqn 3.37

        while fail < max_fail:
            LHS   = (alpha + lam) * eye(r) + M @ T        # LHS of eqn 3.36
            delta = solve(LHS, RHS)
            new_y = y + delta
            err   = chi_sq(Aw, bw, mu, U, new_y)
            if isnan(err) or err_list[-1] < err:
                lam  *= up
                fail += 1
            else:
                lam = 1e-3
                break

        if fail == max_fail:
            print(' --- reached max fail! -- ')
            print('round', i, 'max iter', max_iter,
                  'abs(err_list[-2] - err) / err < 1e-5',
                  abs(err_list[-2] - err) / err)
            break
        else:
            err_list.append(err)
            y = new_y
            if abs(err_list[-2] - err) / err < 1e-5:
                break

    return mu * exp(U @ y)                                 # eqns (3.29) & (3.33)


@njit
def chi_sq(Aw, bw, mu, U, y):
    # Whitened chi-squared: ||Aw x - bw||^2
    # Original had: (A@x-b).T @ eigvals_inv_matrix @ (A@x-b)
    # In the whitened system eigvals_inv_matrix = I, so this is just a dot product.
    x = mu * exp(U @ y)
    r = Aw @ x - bw
    return sum(r.T @ r)
