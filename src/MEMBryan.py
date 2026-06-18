from numpy import shape, reshape, array, diag
from numpy import linspace, zeros, zeros_like, ones, eye, copy
from numpy import sqrt, exp, abs, log10, log
from numpy import isnan, nan, nan_to_num, isfinite, inf, argmax, all
from numpy import sum
from numpy.linalg import svd, eigh, inv, solve, norm

from scipy.linalg import cholesky, solve_triangular


def MEMBryan_solve(A, b, C, mu, alpha_min=1e-3, alpha_max=1e3, Nalpha=19, cutoff=0.1,
                  cond_upperbound=1e10, max_iter=500, max_fail=2000, numerical_zero=1e-14):
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
    #   accepted_arr: array with 0s and 1s indicating whether a particular alpha was included in the averaging.

    # Check the dimension of inputs and make sure everyting agrees
    [Ntau, Nomega] = shape(A)
    b = reshape(b, (Ntau, 1)) # will fail if b has wrong dimension
    mu = reshape(mu, (Nomega, 1)) # will fail if mu has wrong dimension
    if shape(C) != (Ntau, Ntau):
        print('ERROR: C must be (Ntau,Ntau)')

    # --- Cholesky whitening ---
    # We want to solve (Ax-b).T@Cinv@(Ax-b).
    # Let C = L@L.T so that  Cinv=(L.T)^{-1}L^{-1}
    # Then (Ax-b).T@Cinv@(Ax-b) = (L^{-1}(Ax-b)).T(L^{-1}(Ax-b))
    L  = cholesky(C, lower=True)
    Aw = solve_triangular(L, A, lower=True) # L@Aw = A -> Aw = L^[-1]@A
    bw = solve_triangular(L, b, lower=True) # L@bw = b -> bw = L^[-1]@b

    DDL = Aw.T @ Aw # 2nd derivative of GLS: d^2L/(dx)^2 = A.T @ Cinv @ A

    # collect ||A@x -b||^2_Cinv + \alpha S_sj, at different regularization weight (alphas)
    x_arr = [] # solution array
    P_arr = [] # Posterior array
    alphas = 10**linspace(log10(alpha_min), log10(alpha_max), Nalpha)
    for i in range(Nalpha):
        try:
            x = Bryans_alg(Aw, bw, mu, alphas[i], cond_upperbound, max_iter, max_fail)
        except Exception as error:
            print(r'Exception at alpha %.3e' % alphas[i], error)
            continue

        # If everything is working compute the Bayesian posterior
        if all(x >= 0.0) and all(isfinite(x)):
            x[x < numerical_zero] = numerical_zero

            # Compute Shannon-Jaynes entropy, eqn 3.13
            S = sum(x - mu - x*log(x/mu))

            # Compute Likelihood ||A@x -b||^2_Cinv, eqn 3.19
            res = Aw@x - bw
            Lval = 0.5*norm(res)**2

            # compute Gull term sqrt(x)*A.T@Cinv@A*sqrt(x) epn 3.20
            tmp = eye(Nomega)*sqrt(x)
            LAMBDA = tmp@DDL@tmp
            eigvals, _ = eigh(LAMBDA)
            Gull_term = 0.5 * sum( log(alphas[i]/(alphas[i] + eigvals)) )

            x_arr.append(x[:, 0]) # store solutions
            P_arr.append([alphas[i], alphas[i]*S, -Lval, Gull_term]) # store posterior vals

        else:
            print('Invalid solution at alpha', alphas[i])

    x_arr = array(x_arr) # convert to solution list to array
    P_arr = array(P_arr) # convert to posterior list to array
    P_arr = nan_to_num(P_arr, nan=-inf) # zero out any non-sense posterior vals
    objfxn = sum(P_arr[:, 1:], axis=1) # compute the Bayesian posterior
    max_index = argmax(objfxn) # find the posterior max

    # conduct posterior averaging
    accepted_arr = zeros((len(P_arr),))
    x_avg = zeros_like(x_arr[0])
    xsq_avg = zeros_like(x_arr[0])
    P_tot = 0.0
    for i in range(len(P_arr)):
        weight = exp(objfxn[i] - objfxn[max_index])
        if weight > cutoff:
            x_avg += x_arr[i]*weight
            xsq_avg += (x_arr[i]*x_arr[i])*weight
            P_tot += weight
            accepted_arr[i] = 1

    x_avg /= P_tot
    xsq_avg /= P_tot
    x_var = reshape(xsq_avg - x_avg*x_avg, (Nomega,1))
    x_avg = reshape(x_avg, (Nomega,1))

    return x_avg, x_var, x_arr, P_arr, accepted_arr


def Bryans_alg(Aw, bw, mu, alpha, cond_upperbound, max_iter, max_fail):
    # Bryan's Algorithm published - https://link.springer.com/article/10.1007/BF02427376
    # This code follows the convention of Asakawa et al. https://arxiv.org/abs/hep-lat/0011040
    # However, we have used a Cholesky decomposition to scale/rotate A and b.
    # This avoids extra FLOPS to preserve the inverse singular values.
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

    # Parse arguments
    [Ntau, _] = shape(Aw)
    V0, S, Uh = svd(Aw); U0 = Uh.T

    # conditioning
    if cond_upperbound < 0.0:
        r = Ntau
    else:
        for i in range(Ntau):
            if S[0] / S[i] < cond_upperbound:
                r = i + 1
            else:
                break
    V = copy(V0[:, :r])
    U = copy(U0[:, :r])
    S = diag(S[:r])

    M = S@S # simplified M

    y = zeros((r, 1)) # start the optimization at the default model
    err_list = [chi_sq(Aw, bw, mu, U, y)] # compute error of default model

    # Iterate from default model to the unique solution.
    up = 2.0
    i = 0
    while i < max_iter:
        i+=1 # this index keeps track of the number of update iteration
        lam=1e-3; # this is the Levenberg-Marquardt (LM) parameter, it's like a learning rate.
        fail=0; # this index tracks the # of adjustments made to the LM parameter this iteration

        # ------------------------------------------------------- #
        # Compute RHS of eqn 3.36, where b has been relabeled `y' #
        # ------------------------------------------------------- #
        x = mu*exp(U@y) # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy
        DL = Aw@x - bw # derivative of Likelihood (L) w.r.t. Arot@x
        g = S@V.T@DL # compute g from 3.34
        RHS = -alpha*y - g # compute the RHS of eqn 3.36

        T = U.T@diag(x.flatten())@U # pre-compute T from eqn 3.37
        # Attempt update
        while fail < max_fail:
            LHS = (alpha + lam)*eye(r) + M@T
            delta = solve(LHS, RHS)
            new_y = y + delta

            err = chi_sq(Aw, bw, mu, U, new_y)
            if isnan(err) or err_list[-1] < err:
                lam *= up
                fail += 1
            else:
                lam = 1e-3
                break

        if fail == max_fail:
            print("reached max fail")
            break
        else: # sucessful update
            err_list.append(err)
            y = new_y
            if abs(err_list[-2] - err) / err < 1e-5:
                break

    return mu*exp(U@y)


def chi_sq(Aw, bw, mu, U, y):
    x = mu*exp(U@y)
    r = Aw@x - bw
    return sum(r.T@r)
