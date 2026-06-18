import time
from numba import njit
from numpy import shape, reshape, array, diag, concatenate
from numpy import isfinite, isnan, all, any, argmin, argmax, nanargmax, argsort, sort, where
from numpy import linspace, arange, zeros, zeros_like, ones, eye, copy, tile
from numpy import sqrt, exp, abs, log10, log, pi, nan, nan_to_num, inf
from numpy import sum, average, nanprod

# error handling
from numpy import errstate;
from numpy.linalg import LinAlgError

# initialize kernel subfunction
from numpy import cumsum, amin, amax, diff, interp
from scipy.ndimage import gaussian_filter1d

from scipy.optimize import nnls # solve the coeffs
from numpy.linalg import inv # compute Cinv in chi-square
from numpy.random import default_rng # Needed for Metropolis-Hastings update and Langevin proposal
from numpy.linalg import cholesky, solve, eigh, norm, cond # langevin proposal function
from numpy import maximum, minimum, outer # Langevin_proposal function

from numpy.fft import fft, ifft # compute integrated autocorrelation time
from numpy import conjugate # compute integrated autocorrelation time
from scipy.optimize import least_squares # chi2 kink algorithm

import matplotlib.pyplot as plt

def RSOM_solve(A0, b0, C, mu, omegas, taus,
               beta_min=1e0, beta_max=1e4, Nbeta=4, step_size=1.0,
               Npt=10000, Nmh=100, burn_in=100, burn_in_cycle=None, Nsamples=1000,
               genetic_flag=0, p_reset=0.005,
               lambda_min=1e-6, lambda_max=0.5,
               Nkernels_min=1, Nkernels_max=10, Nkernels_step=1,
               detailed_balance=False, Laplace_transform=False,
               alpha_min=1e-3, alpha_max=1e3, Nalpha=31, NC2K=10,
               seed=15485863, grad_scale=1e-7
              ):
    # This code solves ||A@x-b||^2_{Cinv}, by expressing x = K@c
    # K is optained by dictionary learning and stochastic optimization
    # c is obtained by non-negative least squares solution to ||AKc-b||
    #
    # If provided a default model is provided, then the coefficients are obtained via
    #   ||A@K@c-b||^2_{Cinv} + \alpha ||CDF[x - mu]||^2.
    # The CDF regularization is choosen because this cost fxn has a optimum obtained via matrix inversion.
    # The reg. weight alpha is selected via chi-square kink algorithm, becuase the logistic fit is cheap.
    # Note, we modify the chi-square kink alg to select less regularized solutions than recommended.
    #
    # Input:
    #   A: transformation to be inverted
    #   b: data
    #   C: data's convariance matrix.
    #   mu: default model or prior.
    #   omegas: grid of points where x(omega_j) is evaluated.
    #   taus: grid of points where b(tau_i) is evaluated.
    #   beta_min: beta value of hottest walker (beta = 1/T)
    #   beta_max: beta value of coldest walker (beta = 1/T)
    #   Nbeta: Number of walkers (beta = 1/T)
    #   step_size: scales the drift and noise terms in the Langevin-based MH proposal
    #   Npt: maximum number of parallel tempering attempts
    #   Nmh: number of MH update attempts per parallel tempering
    #   burn_in: Number of parallel tempering steps till data collection.
    #   burn_in_cycle: Turn on MH accept/reject every burn_in_cycle
    #                  default - None, i.e., never turn off MH).
    #   Nsamples: number of stochastic samples to accept after burn-in.
    #   genetic_flag: enables genetic elitest step in MH parallel tempering optimization
    #   p_flag: determines probability of elitest step after burn-in
    #   lambda_min=1e-8, lambda_max=0.5 : conditioning the Hessian in the Langevin optimizer
    #   Nkernels_min=1, Nkernels_max=10, Nkernels_step=1, # kernel construction
    #   detailed_balance:
    #   Laplace_transform:
    #   alpha_min: smallest alpha value to test.
    #   alpha_max: largest alpha value to test.
    #   Nalpha: number of points in the alpha grid.
    #   NC2K: number of points to average over in chi-sq kink alg
    #   seed: initialization of the rng for reproducability
    #   seed: tells the optimizer what gradient level to turn on noise
    #
    # Output:
    #     x_avg: MEM estimate
    #     xsq_avg - x_avg*x_avg: the variance of MEM estimate
    #     x_arr: arr of solution (one for each alpha)
    #     P_arr: array of unnormalized Bayesian Posterior weighting function
    #     accepted_arr: array with 0s and 1s indicating whether a particular alpha was included in the averaging.
    # There are many stochastic optimization methods, but this is our SOM.

    # print arguments for the sake of prosperity
    print('step_size', step_size, 'beta_min', beta_min, 'beta_max', beta_max, 'Nbeta', Nbeta)
    print('Npt', Npt, 'Nmh', Nmh, 'burn_in', burn_in, 'Nsamples', Nsamples)
    print('genetic_flag', genetic_flag, 'p_reset', p_reset)

    # Check the dimension of inputs
    [Ntau, Nomega] = shape(A0) # Ntau rows and Nomega cols
    b = reshape(b0, (Ntau, 1)) # make sure that data b is the right shape
    mu = reshape(mu, (Nomega, 1)) # make sure that Bayesian prior is the right shape.

    # Whiten the solution
    L_Cov = cholesky(C) # C = L@L.T --> (Ax-b).T@(L@L.T)^{-1}@(Ax-b) = ||L.T^{-1}@(Ax-b))||_2
    A = solve(L_Cov.T, A0) # L.T@Aw = A --> Aw = L.T^{-1}@A
    b = solve(L_Cov.T, b) # L.T@bw = b --> bw = L.T^{-1}@b
    step_size *= 0.01 # reduce the step size, Langevin proposal will cautiously allow original step_size.

    if (Laplace_transform):
        print('constructing regression matrix via analytic experssion')
    else:
        print('constructing regression matrix via numeric transform (matmul A@K)')

    chisq_arr = []
    RSOM_chisq_arr = []
    sol_arr = []
    var_arr = []
    # ---------------------------------------------------- #
    # Conduct dimensionality scan for dictionary learning.
    # ---------------------------------------------------- #
    Nkernels_max += 1 # make the scan include the argument
    kernel_grid = range(Nkernels_min, Nkernels_max, Nkernels_step)
    for Nkernels in kernel_grid:
        print('number of kernels', Nkernels)

        tmp = stoch_opt(A, b, mu, omegas, taus, \
                        alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha, NC2K=NC2K, \
                        beta_min=beta_min, beta_max=beta_max, Nbeta=Nbeta, step_size=step_size, \
                        Npt=Npt, Nmh=Nmh, burn_in=burn_in, burn_in_cycle=burn_in_cycle, \
                        Nsamples=Nsamples, \
                        genetic_flag=genetic_flag, p_reset=p_reset, \
                        lambda_min=lambda_min, lambda_max=lambda_max, \
                        Nkernels=Nkernels, \
                        detailed_balance=detailed_balance, \
                        Laplace_transform=Laplace_transform, \
                        seed=seed+Nkernels, grad_scale=grad_scale, \
                        A0=A0
                       )
        x_RSOM, x_RSOM_var, RSOM_proposedsolutions, RSOM_chisq = tmp
        sol_arr.append(x_RSOM)
        var_arr.append(x_RSOM_var)
        x_RSOM  = reshape(x_RSOM, (-1, 1))
        res = A@x_RSOM - b
        RSOM_chisq_arr.append(average(RSOM_chisq))
        chisq_arr.append(norm(res)**2/Ntau)

    chisq_arr = array(chisq_arr)
    print('average chisq_arr of fits', RSOM_chisq_arr)
    print('chisq_arr of averaged solutions', chisq_arr)
    sol_arr = array(sol_arr)
    var_arr = array(var_arr)

    # --------------------------------------------- #
    # Enforce sparsity through dictionary learning! #
    # --------------------------------------------- #
    # If 1 Gaussian already good enough then choose it
    # Find first dimension where chi_sq < 1, if no value <1 then return the last element of array
    indices = where(chisq_arr < 2)[0]
    choice_1 = indices[0] if len(indices) > 0 else len(chisq_arr)-1

    # Find dimension where the relative difference is largest.
    rel_diff = abs(chisq_arr[:-1] - chisq_arr[1:]) / chisq_arr[1:]
    # first +1 is to account for the finite diference shifting index, NOPE! second +1 to move slightly past the biggest drop.
    choice_2 = nanargmax(rel_diff) + 1
    print('RSOM DIAGNOSTIC: Nkernels needed for chi-sq < 2', Nkernels_min + choice_1,
          'RSOM DIAGNOSTIC: Nkernels w/ largest relative diff', Nkernels_min + choice_2)

    if chisq_arr[0] < 2:
        choice = 0
        print('1 Gaussian already sufficient (chi^2 < 1)')
    else:
        choice = max(choice_1, choice_2)

    # ------------------------------------------------- #
    # plot the dictionary learning selection procedure. #
    # ------------------------------------------------- #
    plt.figure()
    plt.plot(kernel_grid, chisq_arr, label=r'$\tilde{\chi}^2$')
    plt.plot(kernel_grid[1:], rel_diff, label=r'rel. diff in $\tilde{\chi}^2$')
    plt.axvline(Nkernels_min + choice, label=rf'Selected $N_c$', color='red')
    plt.axhline(1.0, label=r'$\tilde{\chi}^2 = 1$', color='black', ls= '--')
    #plt.axhline(0.634, label=r'Conjugate Gradient $\chi^2$', color='black')
    plt.ylabel(r'$\tilde{\chi}^2 = 1/N_{\tau} ((b - R c) / \delta b)^2$', fontsize=14)
    plt.xlabel(r'Number of kernels $N_c$', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.legend(fontsize=14)
    plt.yscale('log')
    plt.title(r'Selecting $N_c$ to enforce sparsity', fontsize=16)
    plt.savefig('imgs/dictionarylearning_MSE.pdf', bbox_inches='tight', format='pdf')

    x_sol = copy(reshape( sol_arr[choice], (-1,1)))
    x_var = copy(reshape( var_arr[choice], (-1,1)))
    return x_sol, x_var, sol_arr, chisq_arr, choice


def stoch_opt(A, b, mu, omegas, taus, \
              alpha_min, alpha_max, Nalpha, NC2K, \
              beta_min, beta_max, Nbeta, step_size,
              Npt, Nmh, burn_in, burn_in_cycle, \
              Nsamples, \
              genetic_flag, p_reset, \
              lambda_min, lambda_max, \
              Nkernels, \
              detailed_balance, \
              Laplace_transform, \
              seed, grad_scale, A0
             ):
    # Check the dimension of inputs
    [Ntau, Nomega] = shape(A) # Ntau rows and Nomega cols (remember that A and b are rotated!)
    dw = omegas[1] - omegas[0]

    # Initialize a Generator with a fixed seed for reproducibility
    rng = default_rng(seed=seed)

    # Initialize the temperatures of the Walker
    betas = 10**linspace(log10(beta_min), log10(beta_max), Nbeta, endpoint=True)

    # Initialize the regularization grid
    #alphas = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha, endpoint=True) # used for chi-sq kink alg

    # Initialize the kernels
    theta_arr = zeros((Nbeta, 2*Nkernels))
    best_chi2_arr = zeros((Nbeta,));
    walker_chi2_arr = zeros((Nbeta,));
    for i in range(Nbeta):
        centers, sigmas = select_kernels_via_rng(omegas, Nkernels, rng)

        # Realize kernels to compute the initial kernel coefficients
        E = kernel_matrix(omegas, taus, centers, sigmas, detailed_balance)
        R = A@E

        # compute kernel coefficients via regularized optimization
        #kernel_coeff = compute_kernel_coeff_via_C2K(R, E, b, mu, alphas, NC2K)
        # compute kernel coefficients without regularization
        kernel_coeff = reshape(nnls(R, b)[0],(-1,1))

        # Compute the inital chi-square
        res = R@kernel_coeff - b
        best_chi2 = norm(res)**2/Ntau # reduced chi-sq
        best_chi2_arr[i] = best_chi2
        walker_chi2_arr[i] = best_chi2

        # store the compute theta values
        theta = pack_theta(centers, sigmas, omegas[-1])
        theta_arr[i] = theta

    # Per-beta cache for (grad, H): avoids recomputing the expensive Jacobian
    # when a Metropolis-Hastings proposal is rejected and theta is unchanged.
    # Set to None to signal a cache miss; invalidated on acceptance, PT swap, or genetic reset.
    grad_cache = [None] * Nbeta
    H_cache    = [None] * Nbeta

    # stochastic optimization of the chi-square
    collected_x_arr = [];
    collected_chi2_arr = [] # collect the chi-sq on MH acceptance step
    collected_Rcond_arr = [] # collect the condition number kappa(R) on MH acceptance step
    print('burn_in:', burn_in, 'betas', betas)
    for pt_iter in range(Npt + burn_in):
        #start_time = time.time()
        if (burn_in_cycle == None):
            collect_everything = False
        elif ( pt_iter > burn_in and pt_iter % burn_in_cycle == 0 ):
            # every 10 pt cycles after burn_in then accept everything
            collect_everything = True
        else:
            collect_everything = False

        # Step 1: conduct Nmh Metropolis-Hastings updates at each beta
        for beta_index in reversed(range(Nbeta)):
            for mh_iter in range(Nmh):
                # 1a. Propose theta update
                theta = theta_arr[beta_index]

                # Compute gradient or reuse cached result if previous update was rejected.
                if grad_cache[beta_index] is None:
                    result = E_coeff_chi2_grad_H_analytic(A, b, omegas, taus,
                                              theta,
                                              detailed_balance,
                                              Laplace_transform)
                    if result[0] is None:
                        continue
                    _, _, _, grad, H = result
                    grad_cache[beta_index] = grad
                    H_cache[beta_index]    = H
                else:
                    grad = grad_cache[beta_index]
                    H    = H_cache[beta_index]

                # Generate via Langevin proposal, i.e., include gradient info to increase sampling efficiency
                L_Hessian = cholesky_with_damping(H, betas[beta_index], beta_max, lambda_min, lambda_max)
                theta_prop = langevin_proposal(theta, grad, L_Hessian, betas[beta_index], beta_max, step_size, rng, grad_scale)

                # 1b. Evaluate proposal
                # parse current centers and sigmas and make sure the values are sane.
                current_centers, current_sigmas = unpack_theta(theta, omegas[-1])
                proposal_centers, proposal_sigmas = unpack_theta(theta_prop, omegas[-1])
                if any(proposal_sigmas < dw):
                    continue # skip any proposal that suggests a width less than the omega grid spacing.

                # compute the evaluation matrix, kernel coeff, and the chi-sq
                result_prop = E_coeff_chi2_grad_H_analytic(A, b, omegas, taus,
                                               theta_prop,
                                               detailed_balance, Laplace_transform,
                                               chi2_flag=True)

                # Check if the computation was successful.
                if result_prop[0] is None:
                    continue
                else:
                    E, kernel_coeff, proposal_chi2, _, _ = result_prop

                # Diagnostic is the algorithm is not working!
                #if(betas[-1] == betas[beta_index]):
                #    print('RSOM DIAGNOSTIC: beta', beta_index, betas[beta_index], 'proposal (coeff, center, sigma)', kernel_coeff, proposal_centers, proposal_sigmas)

                # 1c: Metropolis Hastings (MH) update
                delta_chi2 = proposal_chi2 - walker_chi2_arr[beta_index]
                #print('RSOM DIAGNOSTIC: final chi2, initial chi2', proposal_chi2, walker_chi2_arr[beta_index], 'delta chi2', delta_chi2)
                if ( (rng.uniform() < exp(-betas[beta_index]*delta_chi2)) or collect_everything):
                    #print('pt, beta, mh', pt_iter, beta_index, mh_iter, 'RSOM: walker_chi2_arr (before update)', walker_chi2_arr)

                    # wait for burn_in (or skip burnin if chi-sq <1),
                    # and check if proposal is below threshold and store the result.
                    if ( ((pt_iter >= burn_in) or (proposal_chi2 < 1)) and (proposal_chi2 < 1.5*amin(walker_chi2_arr)) ): 
                        #print('RSOM DIAGNOSTIC: accepted nth val', len(collected_x_arr) )
                        collected_x_arr.append(E@kernel_coeff)
                        collected_chi2_arr.append( proposal_chi2 )
                        collected_Rcond_arr.append( cond(A0@E) ) #( Rcond )
                        # check if you have stored the requested number of samples, loop over betas.
                        if( len(collected_chi2_arr) == Nsamples ):
                            break

                    walker_chi2_arr[beta_index] = proposal_chi2
                    theta_arr[beta_index] = theta_prop
                    # theta changed, invalidate this walker's grad/H cache
                    grad_cache[beta_index] = None
                    H_cache[beta_index]    = None

            if( len(collected_chi2_arr) == Nsamples ):
                break # exit MH loop

        if( len(collected_chi2_arr) == Nsamples ):
            break # exit walker loop

        # Step 2: Parallel tempering
        print('RSOM DIAGNOSTIC: pt_iter', pt_iter, 'collect_everything', collect_everything, 'num_collected', len(collected_chi2_arr), 'chi2_arr[-10:]', walker_chi2_arr[-10:])
        #print('RSOM DIAGNOSTIC: Pre Parallel Tempering: chi2_arr', walker_chi2_arr)
        for beta_index in range(Nbeta-1):
            delta_beta = betas[beta_index] - betas[beta_index+1]
            delta_chi2 = walker_chi2_arr[beta_index] - walker_chi2_arr[beta_index+1]
            rate = exp(delta_beta*delta_chi2)
            if rng.uniform() < rate:
                #print('RSOM DIAGNOSTIC: accepted a PT switch in walker', beta_index, 'chi2_arr after switch', walker_chi2_arr)
                # swap positions in optimization space
                tmp = copy(theta_arr[beta_index])
                theta_arr[beta_index] = copy(theta_arr[beta_index+1])
                theta_arr[beta_index+1] = tmp
                # swap chi2
                tmp = copy(walker_chi2_arr[beta_index])
                walker_chi2_arr[beta_index] = copy(walker_chi2_arr[beta_index+1])
                walker_chi2_arr[beta_index+1] = tmp
                # thetas swapped – invalidate both walkers' caches
                grad_cache[beta_index]   = None
                grad_cache[beta_index+1] = None
                H_cache[beta_index]      = None
                H_cache[beta_index+1]    = None

        # step 3: Respawn poorly performing walkers
        if (rng.uniform() < p_reset) and (pt_iter > burn_in) and (genetic_flag):
            for i in range(Nbeta-1):
                # gaurantee that the last walker is saved, otherwise
                # reset all walkers that are not fitting the data (chi-sq < 1).
                if (walker_chi2_arr[i] < 1):
                    continue
                centers, sigmas = select_kernels_via_rng(omegas, Nkernels, rng)
                print('initial centers', centers, 'initial sigmas', sigmas)

                # Realize kernels to compute the initial kernel coefficients
                E = kernel_matrix(omegas, taus, centers, sigmas, detailed_balance)
                R = A@E

                # compute kernel coefficients via regularized optimization
                #kernel_coeff = compute_kernel_coeff_via_C2K(R, E, b, mu, alphas, NC2K)
                # compute kernel coefficients without regularization
                kernel_coeff = reshape(nnls(R, b)[0],(-1,1))

                # Compute the inital chi-square
                res = R@kernel_coeff - b
                walker_chi2_arr[i] = norm(res)**2/Ntau

                # store the compute theta values
                theta = pack_theta(centers, sigmas, omegas[-1])
                theta_arr[i] = theta

            # all thetas changed, flush the entire grad/H cache
            grad_cache = [None] * Nbeta
            H_cache    = [None] * Nbeta
        #print('pt iter time', time.time() - start_time)

    print('RSOM DIAGNOSTIC: iteration', pt_iter, 'number of accepted', len(collected_chi2_arr))
    collected_x_arr = array(collected_x_arr)
    collected_chi2_arr = array(collected_chi2_arr)
    collected_Rcond_arr = array(collected_Rcond_arr)

    if (len(collected_chi2_arr) > 1):
        # compute integrated autocorrelation time (IACT) to estimate how many independent samples
        iact_list = [];
        c_value = 5.0
        for i in range(Nomega):
            #iact, acf_full, t_hist, tau_int_hist, break_t, break_reason = get_iact(collected_x_arr[:,i].flatten(), c=c_value)
            iact, _, _, _, _, _ = get_iact(collected_x_arr[:,i].flatten(), c=c_value)
            iact_list.append(iact)
        iact_avg = average(array(iact_list))
        eff_N_samples = len(collected_x_arr)/iact_avg # effective number of samples
        print('RSOM DIAGNOSTIC: iact avg and eff_N_samples', iact_avg, eff_N_samples) #iact_list)

        # compute mean
        x_avg_geometric = geometric_average(collected_x_arr, axis=0)
        # compute uncertainty
        x_avg = average(collected_x_arr, axis=0)
        xsq_avg = average(collected_x_arr**2, axis=0)
        x_var = (xsq_avg - x_avg*x_avg)/eff_N_samples
    else:
        x_avg = zeros((Nomega,1))
        x_var = zeros((Nomega,1))

    print('RSOM DIAGNOSTIC: Rcond list', collected_Rcond_arr[:10], collected_Rcond_arr[-10:])
    print('RSOM DIAGNOSTIC: Average conditioning of R', average(collected_Rcond_arr))
    print('RSOM DIAGNOSTIC: Geometric avg conditioning of R', geometric_average(collected_Rcond_arr))
    return x_avg[:,0], x_var[:,0], collected_x_arr, collected_chi2_arr
    #return x_avg_geometric[:,0], x_var[:,0], collected_x_arr, collected_chi2_arr

@njit(cache=True)
def select_kernels_via_rng(w, Nkernels, rng):
    # Select distribution of kernel centers based on where D(w) has most mass.
    #
    # Parameters:
    #     w: omega grid array.
    #     Nkernels: Number of kernels to initialize.
    #     rng: the random number generator
    #
    # Returns:
    w_length = w[-1] - w[0]
    dw = w[1] - w[0]
    centers = w[0] + (0.95*w_length - w[0])*sort(rng.uniform(size=Nkernels)) # uniform kernels over interval
    sigmas = 2*dw + (0.8*w_length - 2*dw)*rng.uniform(size=Nkernels) # 5x bigger than grid spacing - 80% of total domain
    #sigmas = maximum(2*dw, sigmas) # cannot be thinner than then twice grid (Nyquist)!
    return centers, sigmas


@njit(cache=True)
def langevin_proposal(theta, grad, L, beta, beta_max, step_size_0, rng, grad_scale=1e-7):
    # grad_scale essentially tells the optimizer what to consider larger or small gradient.
    grad_norm = norm(grad)
    #if (beta == beta_max):
    #    print('gradient norm', grad_norm)
    alpha_grad = grad_norm / (grad_norm + grad_scale) # large grad alpha_grad -> 1
    alpha_beta = beta / beta_max  # large beta -> alpha_beta -> 1
    alpha = alpha_grad * alpha_beta
    # (1 - alpha) supresses when grad is large and beta is large
    # (1 + alpha) enhances when grad is large and beta is large

    step_size = step_size_0 * (1 + 99*alpha_grad) # enhance step size [1x-100x] when grad large

    # --- Drift ---
    v    = solve(L, -grad)        # v = L^{-1}(-g)
    drift = solve(L.T, v)         # drift = L^{-T} v = -H^{-1} g
    if norm(drift) > step_size:
        drift = _lm_trust_region_njit(grad, L, step_size)


    # --- Preconditioned MALA noise ---
    noise = rng.normal(size=len(theta))
    noise_scaled = solve(L.T, noise) # noise ~ N(0, H^{-1})
    noise_scaled *= (2.0 * step_size / beta) ** 0.5 # Langevin scaling
    noise_scaled *= (1.0 - alpha) # suppress noise for cold walker or large grad.

    # --- Single trust-region clip on combined step ---
    step = drift + noise_scaled
    #step_norm = norm(step)
    #if step_norm > step_size:
    #    step *= step_size / step_norm
    #    #if (beta == beta_max):
    #    #    print('cutting step with trust region')
    #print('langevin proposal (drift, noise_scaled)', drift, noise_scaled)
    return theta + step

@njit(cache=True)
def _lm_trust_region_njit(grad, L, delta, max_iter=20, tol=1e-8):
    n = len(grad)
    H = L @ L.T
    lo = 0.0
    hi = norm(grad) / delta + 1.0
    p = zeros(n)
    for _ in range(max_iter):
        lam = (lo + hi) / 2.0
        A = H + lam * eye(n)
        p = solve(A, -grad)
        pnorm = norm(p)
        if abs(pnorm - delta) < tol * delta:
            return p
        if pnorm > delta:
            lo = lam
        else:
            hi = lam
    return p

#@njit(cache=True)
def cholesky_with_damping(H, beta, beta_max, lambda_min, lambda_max, growth=10.0, max_tries=12):

    dim = H.shape[0]

    # Smooth interpolation of damping
    beta_relative = beta / beta_max
    lambda_reg = lambda_min + (lambda_max - lambda_min) * (1.0 - beta_relative**2)

    I = eye(dim)
    for _ in range(max_tries):
        # Regularized Hessian
        Heff = (1.0 - lambda_reg) * H + lambda_reg * I
        Heff = 0.5 * (Heff + Heff.T) # Force symmetry

        try:
            return cholesky(Heff)

        except LinAlgError:
            # failed inversion, increase damping
            lambda_reg *= growth

    print('ERROR: Hessian inversion unstable using identity')
    return I

@njit(cache=True)
def kernel_matrix(omegas, taus, centers, sigmas, detailed_balance):
    # This function takes in a collection of mean and stdev
    # then creates a matrix of the Gaussians
    # Inputs:
    #  omegas, beta, centers, sigmas
    # returns:
    #  regression matrix R with shape [N_tau, N_coeff]
    N_centers = len(centers)
    N_sigmas = len(sigmas)
    if (N_centers != N_sigmas):
        print('ERROR: N_centers != N_sigmas in regression matrix call')
    if (any(sigmas == 0)):
        print('ERROR: A sigma is zero!')
        print('centers', centers)
        print('sigmas', sigmas)
        sigmas[sigmas == 0] = omegas[1] - omegas[0]

    E = zeros((len(omegas), N_centers))
    for i, center, sigma in zip(range(N_centers), centers, sigmas):
        #E[:,i] = 1/sigma/sqrt(2*pi)*exp(-0.5*((omegas - center)/sigma)**2)
        #if(detailed_balance):
        #    E[:,i] += exp(taus[-1]*omegas)/sigma/sqrt(2*pi)*exp(-0.5*((-omegas - center)/sigma)**2)
        E[:,i] = exp(-((omegas - center)/sigma)**2)
        if(detailed_balance):
            E[:,i] += exp(taus[-1]*omegas)*exp(-((-omegas - center)/sigma)**2)
    return E

def E_coeff_chi2_grad_H_analytic(A, b, omegas, taus, theta,
                                 detailed_balance, Laplace_transform,
                                 chi2_flag=False):

    forward = forward_model(A, b, omegas, taus,
                            theta, detailed_balance, Laplace_transform)

    if forward is None:
        return None, None, None, None, None

    E, R, kernel_coeff, residual, chi2 = forward


    if chi2_flag:
        return E, kernel_coeff, chi2, None, None

    grad, H = grad_H_from_forward(A, omegas, taus, theta,
                                  E, R, kernel_coeff, residual,
                                  detailed_balance, Laplace_transform)

    return E, kernel_coeff, chi2, grad, H

def forward_model(A, b, omegas, taus, theta, detailed_balance, Laplace_transform):
    Ntau = len(b)
    omega_max = omegas[-1]

    centers, sigmas = unpack_theta(theta, omega_max)

    E = kernel_matrix(omegas, taus, centers, sigmas, detailed_balance)

    if Laplace_transform:
        R = regression_matrix(taus, centers, sigmas, detailed_balance)
    else:
        R = A @ E

    try:
        kernel_coeff = reshape(nnls(R, b)[0], (-1,1))
        residual = R@kernel_coeff - b
        chi2= norm(residual)**2/Ntau
        #model = R @ kernel_coeff
        #residual = model - b
        #chi2 = (residual.T @ residual)[0,0] / Ntau
    except:
        return None

    return E, R, kernel_coeff, residual, chi2


# Compute the Gradient + Hessian
def grad_H_from_forward(A, omegas, taus, theta,
                        E, R, kernel_coeff, residual,
                        detailed_balance, Laplace_transform):

    Ntau = len(residual)
    omega_max = omegas[-1]

    centers, sigmas = unpack_theta(theta, omega_max)
    Nkernels = len(centers)
    dim = 2 * Nkernels

    coeffs = kernel_coeff[:, 0]
    sig2 = sigmas**2
    sig3 = sigmas**3
    Jc = zeros((Ntau, Nkernels))
    Js = zeros((Ntau, Nkernels))

    for k in range(Nkernels):
        delta = omegas - centers[k]   # omega - center_k  (main term)

        # Recompute the main Gaussian component explicitly so that the
        # detailed-balance contribution in E[:,k] does not pollute the
        # derivatives (the two terms have different delta arguments).
        #
        # New kernel: E_main = exp(-((omega - center_k)/sigma_k)^2)
        #
        # dE_main/d_center_k = +2*(omega - center_k)/sigma_k^2 * E_main
        # dE_main/d_sigma_k  = +2*(omega - center_k)^2/sigma_k^3 * E_main
        #   (no "-1/sigma" term: the old kernel had that from differentiating
        #    the 1/(sigma*sqrt(2*pi)) normalization, which is absent here)
        E_main_k = exp(-(delta / sigmas[k])**2)

        dE_dcenter_k = 2.0 * delta / sig2[k] * E_main_k
        dE_dsigma_k  = 2.0 * delta**2 / sig3[k] * E_main_k

        if detailed_balance:
            # DB term: E_db = exp(tau_max*omega) * exp(-((-omega - center_k)/sigma_k)^2)
            #                = exp(tau_max*omega) * exp(-((omega + center_k)/sigma_k)^2)
            # Let delta_db = omega + center_k.
            #
            # dE_db/d_center_k = -2*(omega + center_k)/sigma_k^2 * E_db
            # dE_db/d_sigma_k  = +2*(omega + center_k)^2/sigma_k^3 * E_db
            delta_db  = omegas + centers[k]
            E_db_k    = exp(taus[-1] * omegas) * exp(-(delta_db / sigmas[k])**2)

            dE_dcenter_k -= 2.0 * delta_db / sig2[k] * E_db_k
            dE_dsigma_k  += 2.0 * delta_db**2 / sig3[k] * E_db_k

        # Weight by the NNLS coefficient c_k.
        # For sigma, also apply the chain rule for the log parameterization
        #   theta_sigma = log(sigma)  =>  d/d_theta_sigma = d/d_sigma * sigma
        Jc[:, k] = A @ (coeffs[k] * dE_dcenter_k)
        Js[:, k] = A @ (coeffs[k] * sigmas[k] * dE_dsigma_k)

    grad_c = (2.0/Ntau) * (Jc.T @ residual).flatten()
    grad_s = (2.0/Ntau) * (Js.T @ residual).flatten()

    H_c = (2.0/Ntau) * (Jc.T @ Jc)
    H_s = (2.0/Ntau) * (Js.T @ Js)

    # Transform
    Jtransform = zeros((Nkernels, Nkernels))
    u0 = theta[0]
    v  = theta[1:Nkernels]

    sig = 1.0 / (1.0 + exp(-u0))
    dc0_du0 = omega_max * sig * (1.0 - sig)
    Jtransform[:, 0] = dc0_du0

    for k in range(1, Nkernels):
        Jtransform[k:, k] = exp(v[k-1])

    grad_theta_centers = Jtransform.T @ grad_c
    grad_theta = concatenate([grad_theta_centers, grad_s])

    H_theta = zeros((dim, dim))
    H_theta[:Nkernels, :Nkernels] = Jtransform.T @ H_c @ Jtransform
    H_theta[Nkernels:, Nkernels:] = H_s

    return grad_theta, H_theta

@njit(cache=True)
def pack_theta(centers, sigmas, omega_max):

    Nkernels = len(centers)
    theta = zeros(2*Nkernels)

    # invert sigmoid for first center
    theta[0] = log(centers[0] / (omega_max - centers[0]))

    # log spacings
    for i in range(1, Nkernels):
        spacing = centers[i] - centers[i-1]
        theta[i] = log(spacing)

    # log sigmas
    theta[Nkernels:] = log(sigmas)

    return theta


@njit(cache=True)
def unpack_theta(theta, omega_max):

    # parse arguments
    Nkernels = int(len(theta)/2)

    # centers are assumed to lie in first half of theta
    u0 = theta[0]
    v  = theta[1:Nkernels]

    centers = zeros(Nkernels)
    # first center in (0, omega_max)
    centers[0] = omega_max / (1.0 + exp(-u0))  # sigmoid scaling
    # cumulative positive spacings
    for i in range(1, Nkernels):
        centers[i] = centers[i-1] + exp(v[i-1])

    # sigmas are assumed to lie in the second half of theta
    sigmas = exp(theta[Nkernels:2*Nkernels])

    return centers, sigmas


# A helper function to extract the data needed for visualization from the loop
def get_iact(time_series, c=5.0):
    n = len(time_series)

    if n < 2:
        return 0.0, array([]), array([]), array([]), None, None

    mean = average(time_series)
    centered_time_series = time_series - mean

    # Compute ACF using FFT (same as original function)
    n_fft = 1 << (2 * n - 1).bit_length()
    f = fft(centered_time_series, n=n_fft)
    acf = ifft(f * conjugate(f)).real[:n]
    acf_0 = acf[0]
    if acf_0 == 0: # Avoid division by zero if input is constant
        acf = zeros_like(acf)
    else:
        acf /= acf_0

    tau_int_history = [1.0] # Initial tau_int at t=0
    t_history = [0]
    current_tau_int = 1.0

    break_t = None
    break_reason = None # 'acf_negative' or 'c_tau_int_exceeded'

    # The loop from the original integrated_autocorrelation_time function
    for t in range(1, n):
        if acf[t] <= 0:
            break_t = t
            break_reason = 'acf_negative'
            break

        current_tau_int += 2.0 * acf[t]
        tau_int_history.append(current_tau_int)
        t_history.append(t)

        if t > c * current_tau_int:
            break_t = t
            break_reason = 'c_tau_int_exceeded'
            break

    return current_tau_int, acf, array(t_history), array(tau_int_history), break_t, break_reason



# ------------------------------------------- #
# Stuff to include an explicit regularization #
# ------------------------------------------- #
@njit(cache=True)
def regression_matrix(taus, centers, sigmas, detailed_balance):
    # This function takes in a collection of mean and stdev
    # then creates a matrix of the Laplace transformed Gaussians
    # Inputs:
    #  taus, centers, sigmas
    # returns:
    #  regression matrix R with shape [N_tau, N_coeff]
    N_centers = len(centers)
    N_sigmas = len(sigmas)
    if (N_centers != N_sigmas):
        print('ERROR: N_centers != N_sigmas in regression matrix call')

    R = zeros((len(taus), N_centers))
    for i, center, sigma in zip(range(N_centers), centers, sigmas):
        # New kernel: exp(-((omega-center)/sigma)^2).
        # Completing the square in the Laplace integral gives a sigma^2*tau^2/4 coefficient
        # (vs. sigma^2*tau^2/2 for the old exp(-0.5*((omega-center)/sigma)^2) kernel).
        R[:,i] = exp(-center*(taus[-1]-taus) + 0.25*sigma**2*(taus[-1]-taus)**2)
        if(detailed_balance):
            R[:,i] += exp(-center*taus + 0.25*sigma**2*taus**2)
    return R

def compute_kernel_coeff_via_C2K(R, E, b, mu, alphas, NC2K, maxiter=1000):
    # This code solves ||R@coeff-b||^2 + alpha*||CDF(E@coeff)-CDF(mu)||^2 w/ non-negative optimizer
    # at a collection of alpha values then uses the chi2kink selection alg to determine the best alpha val.
    # The closed form solution to this optimization problem is given
    # Mat@x = vec,
    # Mat = R.T@Cinv@R + alpha*E.T*Wdiag@E
    # vec = R.T@Cinv@b + alpha*E.T*Wdiag@mu
    # input:
    # output:
    Ntau = len(b)
    [Nomega, Ncoeffs] = shape(E)
    chi2kink_vals = linspace(2.0,2.5, NC2K) # values recommended by Kaufman and Held

    # declare quantities before the for loop for computational savings.
    W = diag(arange(Nomega, 0, -1))
    Mat_1 = R.T@R
    Mat_2 = E.T@W@E
    vec_1 = R.T@b
    vec_2 = E.T@W@mu

    alpha_arr = [];
    chi2_arr = [];
    for i, alpha in enumerate(alphas):
        try:
            Mat = Mat_1/alpha+Mat_2
            vec = vec_1/alpha+vec_2
            kernel_coeff, r = nnls(Mat, vec[:,0], maxiter=maxiter) # solve the optimization problem
            kernel_coeff = reshape(kernel_coeff, (-1,1))
            x = E @ kernel_coeff # compute associated x
            b_x = R @ kernel_coeff # compute associated Laplace transform of solution
        except Exception as error:
            print(r'RSOM: An exception occurred at $\alpha$: %.3e'%alphas[i], error)
        else:
            if ( not all(isfinite(x)) or any(x<0.) ):
                print('RSOM (chi-sq grid): invalid solution there are NaNs, infs, or negatives in the reconstruction')
            else: # construct the chi-sq curve across alpha
                alpha_arr.append( alpha )
                chi2_arr.append( 1./Ntau*((b_x - b).T@(b_x - b))[0,0])

    # 2) fit logistic growth curve to the chi^2(\alpha).
    chi2_arr = array(chi2_arr)
    x = log10(array(alpha_arr))
    y = log10(array(chi2_arr))
    dy = 0.1*abs(log10(array(chi2_arr)))
    try:
        params, error = fit_logistic( x, y, dy )
        [_, _, c, d] = params
    except Exception as error:
        print(r'could not fit the logistic curve for chi-sq kink', error)
        print('chi-sq arr', array(chi2_arr))
        c, d = 10, 1 # heavily over regularize to throw out the sample.

    # estimate best lambda values according to chi^2-kink
    best_alphas = 10**(c - chi2kink_vals/d)
    best_alphas[best_alphas < amin(alphas)] = amin(alphas)
    best_alphas[best_alphas > amax(alphas)] = amax(alphas)

    # compute solutions at new best values
    coeff_arr = [];
    for i, alpha in enumerate(best_alphas):
        try:
            Mat = Mat_1/alpha+Mat_2
            vec = vec_1/alpha+vec_2
            kernel_coeff, r = nnls(Mat, vec[:,0]) # solve the optimization problem
            kernel_coeff = reshape(kernel_coeff, (-1,1))
            x = E @ kernel_coeff # compute associated x
            b_x = R @ kernel_coeff # compute associated Laplace transform of solution
        except Exception as error:
            print(r'RSOM (best alphas): An exception occurred at $\alpha$: %.3e'%alphas[i], error)
            #print(r'RSOM (best alphas): ', chi2_arr)
        else:
            if ( not all(isfinite(x)) or any(x<0.) ):
                print('RSOM (best alphas): invalid solution there are NaNs, infs, or negatives in the reconstruction')
            else: # construct the chi-sq curve across alpha
                coeff_arr.append( kernel_coeff )

    if not coeff_arr:
        print('coeff was empty')
        coeff_avg = zeros((Ncoeffs,))
    else:
        coeff_avg = average(array(coeff_arr), axis=0)
    return reshape(coeff_avg,(-1,1))

def geometric_average(arr, axis=None):
    if (axis==None):
        return exp(average(log(arr))) #nanprod(arr)**(1.0/len(arr))
    else:
        length = shape(arr)[axis]
        return nanprod(arr**(1.0/length), axis=axis)

# Fit logitic function
def fit_logistic(x,y,dy):
    initial_guess = [0.0, 2.8, 10.0, 0.3]  # Initial guess for (a, b, c, d)
    res = least_squares(logistic_residuals, x0=initial_guess, args=(x, y, dy), loss='soft_l1', f_scale=3.0, max_nfev=10000)
    return res.x, None

def logistic_residuals(params, x, y, dy):
    a, b, c, d = params
    return (logistic_function(x, a, b, c, d) - y)/dy

# Define the logistic function
@njit
def logistic_function(alpha, a, b, c, d):
    return a + b / (1 + exp(-d * (alpha - c)))
