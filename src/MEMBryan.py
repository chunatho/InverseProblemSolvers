from numba import njit
from numpy import shape, reshape, array, diag
from numpy import linspace, zeros, zeros_like, ones, eye, copy
from numpy import sqrt, exp, abs, log10, log
from numpy import isnan, nan, nan_to_num, isfinite, inf, argmax, all
from numpy import sum
from numpy.linalg import svd, eigh, inv, solve
#from scipy.linalg import lu_factor, lu_solve

#@njit(parallel=True)
def MEMBryan_solve(A, b, C, mu, alpha_min=1e-3, alpha_max=1e3, Nalpha = 19, cutoff=0.1,\
                     cond_upperbound = 1e10, max_iter=500, max_fail=2000, numerical_zero=1e-14):
    # This code optimizes ||A@x -b||^2_C + \alpha S_sj using Bryan's algorithm.
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
    #   cond_upperbound: Cut off for maximum condition number of discretized kernel.
    #   max_iter: maximum number of Levenberg Marquart steps to take
    #   max_fail: maximum number of # of adjustments made to the LM parameter in an iteration
    # Output:
    #   x_avg: MEM estimate
    #   xsq_avg - x_avg*x_avg: the variance of MEM estimate
    #   x_arr: arr of solution (one for each alpha)
    #   P_arr: array of unnormalized Bayesian Posterior weighting function
    #   accepted_arr: array with 0's and 1's indicating whether a particular alpha was included in the averaging.

    # Check the dimension of inputs
    [Ntau,Nomega] = shape(A) # Ntau rows and Nomega cols
    b = reshape( b,(Ntau,1) )
    [dim1, dim2] = shape(b) # Ntau rows and 1 cols
    if( dim1 != Ntau or dim2 != 1):
        print("ERROR: b needs to be a vector of dimension (Ntau,1)")
    [dim1, dim2] = shape(C) # Ntau rows and Ntau cols
    if( dim1 != Ntau or dim2 != Ntau):
        print("ERROR: C needs to be a covariance matrix of dimension (Ntau, Ntau)")
    mu = reshape(mu, (Nomega,1))
    [dim1, dim2] = shape(mu) # Nomega rows and 1 cols
    if( dim1 != Nomega or dim2 != 1):
        print("ERROR: mu needs to be a vector of dimension (Nomega,1)")

    x_arr = []; # this list will be converted into an array! x_arr = zeros((Nalpha, Nomega));
    P_arr = []; # this list will be converted into an array! P_arr = zeros((Nalpha, 4));
    #alphas = geomspace(alpha_min, alpha_max, Nalpha)
    Cinv = inv(C)
    DDL = A.T@Cinv@A
    alphas = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha)
    for i in range( Nalpha ):
        try:
            x = Bryans_alg(A, b, C, mu, alphas[i], cond_upperbound, max_iter, max_fail)
        except Exception as error:
            print(r'An exception occurred at $\alpha$: %.3e'%alphas[i], error)
        else:
            if ( all(x >= 0.0) and all( isfinite(x) ) ):
                # Shannon Jaynes Entropy eqn 3.13
                x[ x < numerical_zero ] = numerical_zero # if x is too small the log diverges.
                S = sum( x - mu - x * log( x / mu ) )

                # Gull Term x*A.T@C@A*x epn 3.20
                #for j in range(Nomega):
                #    for k in range(Nomega):
                #        LAMBDA[j,k] = sqrt(x[j])*DDL[j,k]*sqrt(x[k])
                tmp = eye(Nomega)*sqrt(x)
                LAMBDA = tmp@DDL@tmp
                LAMBDA_eigvals, trash = eigh(LAMBDA)
                Gull_term = 0.5*sum( log( alphas[i] / (alphas[i] + LAMBDA_eigvals)) )

                # Compute likelihood eqn 3.19
                res = A@x - b
                L = (0.5*res.T @ Cinv @ res )[0,0] # eqn 3.6
                PalphaHm= 1 # Laplace's Rule;
                #PalphaHm= 1/alpha # Jeffery's Rule;
                x_arr.append(x[:,0])
                P_arr.append([alphas[i], alphas[i]*S, -L, Gull_term])
            else: # determine value of objective function associated to the solution
                print('there is at least one nan, inf, or negative in the reconstruction')
    x_arr = array(x_arr);
    P_arr = array(P_arr);
    P_arr = nan_to_num(P_arr, nan==-inf)
    accepted_arr= zeros( (len(P_arr),) )

    # step 2: compute solution averaging over alpha (pg 12)
    objfxn = sum(P_arr[:,1:],axis=1)
    max_index = argmax( objfxn )
    x_avg = zeros_like(x_arr[0])
    xsq_avg = zeros_like(x_arr[0])
    P_tot = 0.0
    for i in range( len(P_arr) ):
        # accept or reject alpha based on weighting function
        weight = exp(objfxn[i] - objfxn[max_index])
        if ( weight > cutoff ):
            x_avg += x_arr[i]*weight
            xsq_avg += (x_arr[i]*x_arr[i])*weight
            P_tot += weight
            accepted_arr[i]=1

    x_avg /= P_tot
    xsq_avg /= P_tot
    return x_avg, xsq_avg - x_avg*x_avg, x_arr, P_arr, accepted_arr

#Use only the likelihood function to weight solution (this is what Asakawa does)
            #x_avg += x_arr[i]*exp(P_arr[i,2] - P_arr[maxalpha,2])
            #xsq_avg += (x_arr[i]*x_arr[i])*exp(P_arr[i,2] - P_arr[maxalpha,2])
            #P_tot += exp(P_arr[i,2] - P_arr[maxalpha,2])
            #accepted_arr[i]=1

@njit(cache=True)
def Bryans_alg(A, b, C, mu, alpha, cond_upperbound,  max_iter, max_fail):
    # Bryan's Algorithm published - https://link.springer.com/article/10.1007/BF02427376
    # This code follows the convention of Asakawa et al. https://arxiv.org/abs/hep-lat/0011040
    # Input:
    #   A: matrix to be inverted.
    #   b: data
    #   C: data's convariance matrix.
    #   mu: default model or prior.
    #   alpha: weight of the entropic prior.
    #   cond_upperbound: Cut off for maximum condition number of discretized kernel.
    #      default (1e10) - this default value is set in the MEM Bayesian averaging wrapper function.
    #   max_iter: maximum number of Levenberg Marquart steps to take
    #      default (500) - this default value is set in the MEM Bayesian averaging wrapper function.
    #   max_fail: maximum number of # of adjustments made to the LM parameter in an iteration
    #      default (2000) - this default value is set in the MEM Bayesian averaging wrapper function.
    # Output:
    #     x: solution
    [Ntau,Nomega] = shape(A)
    # Decompose the convariance matrix
    eigvals, R = eigh(C) # C = R Lambda R.T (note Cinv = R lambda^{-1} R.T )
    eigvals_inv_matrix = diag(1./eigvals)

    # Rotate into diagonal covariance space (eqn 3.39)
    Arot = copy(R.T @ A)
    brot = copy(R.T @ reshape(b, (Ntau,1)) )

    # let svd produce A = V@diag(S)@Uh, thus we have to transpose Vh to get V
    # A=V@S@U.T disagrees with the typical A=U@S@V.T, but it is
    # how Asakawa writes it (see the text below eqn 3.30).
    V0,S,Uh = svd(Arot); U0=Uh.T
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
    # use copy to make sure arrays are c ordered for numba
    V=copy(V0[:,:r:1]); U=copy(U0[:,:r:1]); S = diag(S[:r:1])

    # Precompute the M for computational savings (eqn 3.37)
    M = S @ V.T @ (eigvals_inv_matrix) @ V @ S

    y = zeros((r,1), dtype='float64') # start optimization at default model
    err_list=[ chi_sq(Arot, brot, eigvals_inv_matrix, mu, U, y) ];

    #initialize algorithm constants
    up=2.; c0=.1; i=0;
    while (i < max_iter):
        i+=1 # this index keeps track of the number of update iteration
        lam=1e-3; # this is the Levenberg-Marquardt (LM) parameter, it's like a learning rate.
        fail=0; # this index tracks the # of adjustments made to the LM parameter this iteration

        # ------------------------------------------------------- #
        # Compute RHS of eqn 3.36, where b has been relabeled `y' #
        # ------------------------------------------------------- #
        x = mu*exp(U @ y) # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy
        DL = eigvals_inv_matrix @ ( Arot @ x - brot ) # derivative of Likelihood (L) w.r.t. Arot@x
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
            #flag = ( (delta.T @ T @ delta)[0,0] > c0 ) # eqn 3.38
            if ( isnan(err) or err_list[-1] < err): # only update if Chi_sq improves!
            #if ( (delta.T @ T @ delta)[0,0] > c0*sum(m) ): # Asakawa update
                lam*=up;
                fail+=1;
            else: # successful update - reset Lev. Marq. parameter
                lam=1e-3
                break

        # check if fail limit has been reached
        if (fail == max_fail):
            print("reached max fail")
            break
        #update and check for convergence
        else:
            err_list.append(err);
            y=new_y;
            if ( abs(err_list[-2] - err) / err < 1e-5 ):
                break

    x = mu*exp(U @ y); # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy
    return x

@njit
def chi_sq(A, b, eigvals_inv_matrix, mu, U, y):
    x = mu*exp(U @ y) # combining eqns (3.33) a = Uy and (3.29) x = m * exp(a)
    return sum((A@x-b).T @ eigvals_inv_matrix @ (A@x-b))
