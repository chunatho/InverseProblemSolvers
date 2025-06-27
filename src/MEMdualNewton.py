from numpy import shape, reshape, array, diag
from numpy import linspace, zeros, zeros_like, ones
from numpy import sqrt, exp, abs, log10, log, argmax
from numpy import isfinite, inf
from numpy import any, logical_and
from numpy import sum
from numpy.linalg import svd, eigh, inv
print('Loading Dual perspective, make sure you have installed')
print('this package using either >>> pip install DualPerspective')
print('or >>> pip install --force-reinstall DualPerspective')
# to install DualPerspective use:
# >>> pip install DualPerspective
# if you need to update or reinstall Julia use:
# >>> pip install --force-reinstall DualPerspective
from DualPerspective import DPModel, solve, regularize
#from numba import njit

def MEMdN_solve(A, b, C, mu,  alpha_min=1e0, alpha_max=1e8, Nalpha = 19, posterior_cutoff=0.1, cond_upperbound = -1, rtol=1e-8, atol =1e-8, xatol=1e-8, xrtol=1e-8, numerical_zero=1e-16):
    # This code optimizes ||A@x-b||^2_C + \alpha S_sj using Bryan's algorithm.
    # The details are described in Asakawa et al. https://arxiv.org/abs/hep-lat/0011040
    # This function solves Bryan's algorithm for Nalpha points in the interval [alpha_min, alpha_max].
    # Then this function combines the solutions into a single estimate by weighting each solution
    # according to a Bayesian posterior distribution.
    # Input:
    #   A: matrix to be inverted.
    #   b: data
    #   C: data's convariance matrix.
    #   mu: default model or prior.
    #   alpha_min: smallest alpha value to test
    #   alpha_max: largest alpha value to test
    #   Nalpha: number of points in the alpha grid
    #   posterior_cutoff: Within the alpha domain [alpha_min, alpha_max] a given alpha is acceptable if
    #                     P(alpha)/max{P(\alpha)} > cut_off
    #   cond_upperbound: Cut off for maximum condition number of discretized kernel.
    #   rtol: the relative tolerance of the solution ||Ax-b||^2
    #   atol: absolute tolerance for minimum of ||Ax-b||^2
    #   xrtol: relative tolerance for scaling of x
    #   xatol: absolute tolerance for scaling of x
    #
    # Output:
    #     x_avg: MEM estimate
    #     xsq_avg - x_avg*x_avg: the variance of MEM estimate
    #     x_arr: arr of solution (one for each alpha)
    #     P_arr: array of unnormalized Bayesian Posterior weighting function
    #     accepted_arr: array with 0's and 1's indicating whether a particular alpha was included in the averaging.

    # Check the dimension of inputs
    [Ntau,Nomega] = shape(A) # Ntau rows and Nomega cols
    b = reshape( b,(Ntau,1) )
    [dim1, dim2] = shape(b) # Ntau rows and 1 cols
    if( dim1 != Ntau or dim2 != 1):
        print('ERROR: b needs to be a vector of dimension (Ntau,1)')
    [dim1, dim2] = shape(C) # Ntau rows and Ntau cols
    if( dim1 != Ntau or dim2 != Ntau):
        print('ERROR: C needs to be a covariance matrix of dimension (Ntau, Ntau)')
    mu = reshape(mu, (Nomega,1))
    [dim1, dim2] = shape(mu) # Nomega rows and 1 cols
    if( dim1 != Nomega or dim2 != 1):
        print('ERROR: mu needs to be a vector of dimension (Nomega,1)')

    # Instantiate dual problem to solve
    #alphas = geomspace(alpha_min, alpha_max, Nalpha)
    alphas = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha)
    Cinv = inv(C)
    model = DPModel(A, b[:,0], q=mu[:,0], C=C) #, c=one_vector,λ=None)  # Optionally: DPModel(A, b, q=None, C=None, c=None, λ=None)

    # Declare arrays for storing solutions and Bayesian posterior.
    x = zeros((Nomega,1)); # this list will be converted into an array! x_arr = zeros((Nalpha, Nomega));
    x_arr = zeros((Nalpha, Nomega)); # this list will be converted into an array! x_arr = zeros((Nalpha, Nomega));
    P_arr = zeros((Nalpha, 4)); # this list will be converted into an array! P_arr = zeros((Nalpha, 4));
    accepted_arr= zeros((Nalpha,))

    # Step 1: Solve dual problem solution and estimate Bayesian posterior at each alpha 
    for i in range( Nalpha ):
        regularize(model, alphas[i])  # set regularization of optimization problem
        x[:,0] = solve(model)
        if ( np_all(x >= 0.0) and np_all( isfinite(x) ) ):
            # determine value of objective function associated to the solution
            # compute Shannon Jaynes Entropy (eqn 3.13)
            x = lower_bound(x, numerical_zero) # if x is too small the log diverges, so set some lower bound
            S = sum( x - mu - x * log( x / mu ) )
            # compute Gull term (eqn 3.20) - A.T@C@A is the 2nd derivative of L w.r.t. A
            LAMBDA = sqrt(x) *  (A.T@ Cinv @ A) * sqrt(x).T
            LAMBDA_eigvals, trash = eigh(LAMBDA)
            if ( any( (alphas[i] + LAMBDA_eigvals) < 0) ):
                # The magnitude of alpha is so low that the Gull term fails.
                # this failure  precludes the use of this solution
                term = -inf
            else:
                term = 0.5*sum( log( alphas[i] / (alphas[i] + LAMBDA_eigvals)) )
            # compute generalized least squares (eqn 3.19)
            L = (1./2. * ( A@x - b ).T @ Cinv @ ( A@x - b ) )[0,0] # eqn 3.6
            PalphaHm= 1 # Laplace's Rule;
            #PalphaHm= 1/alpha # Jeffery's Rule;
            #x_arr[i] = x.flatten()
            #P_arr[i] = [alphas[i], alphas[i]*S, -L, term]
            x_arr[i] = x.flatten()
            P_arr[i,0] = alphas[i]
            P_arr[i,1] = alphas[i]*S
            P_arr[i,2] = -L
            P_arr[i,3] = term
        else:
            if (not np_all( isfinite(x) ) ):
                print('invalid solution there are NaNs or infs in DSF')
            if ( not np_all(x > 0.0)):
                print('invalid solution there are negatives in DSF')
            P_arr[i,0] = alphas[i]
            P_arr[i,1] = -inf
            P_arr[i,2] = -inf
            P_arr[i,3] = -inf

    # step 2: compute solution averaging over alpha (pg 12)
    objfxn = sum(P_arr[:,1:],axis=1)
    max_index = argmax( objfxn )
    x_avg = zeros_like(x_arr[0])
    xsq_avg = zeros_like(x_arr[0])
    P_tot = 0.0
    for i in range( len(P_arr) ):
        # accept or reject alpha based on weighting function
        weight = exp(objfxn[i] - objfxn[max_index])
        if ( weight > posterior_cutoff ):
            x_avg += x_arr[i]*weight
            xsq_avg += (x_arr[i]*x_arr[i])*weight
            P_tot += weight
            accepted_arr[i]=1

    x_avg /= P_tot
    xsq_avg /= P_tot
    return x_avg, xsq_avg - x_avg*x_avg, x_arr, P_arr, accepted_arr

#@njit(cache=True)
def np_all(x):
    # Numba compatible version of np.all(x, axis=0)
    out = True
    for bool in x.flatten():
        out = logical_and(out, bool)
    return out

#@njit(cache=True)
def lower_bound(x, eps):
    # Numba compatible version of x[ x < eps] = eps.
    for i, val in enumerate(x.flatten()):
        if ( val < eps):
            x[i] = eps
    return x

