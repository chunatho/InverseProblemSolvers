from numba import njit
from numpy import fmax
from numpy import shape, reshape, ones, diag
from numpy.linalg import norm, svd, inv

@njit
def SD_solve( A, b, C, cond_upperbound=-1, tol=1e-6, max_iter=100000, proj_flag=0 ):
    """
    This methods solves the minimizes ||Ax-b||^2_C using the Projected Steepest Descent 
    stopping the algorithm early is a form of regularization.
    # tol: the convergence tolerance.
    # proj_flag: project each steepest descent step into positive function space.
    # Max_It: maximum nuber of gradient descent steps that can be taken.
    INPUTS
     A: is the inversion kernel (shaped [Ntau,Nomega])
     b: the data (shaped [Ntau,1])
     C: the covariance matrix (shaped [Ntau,Ntau])
     tol: sets the stopping condition for the algorithm as norm(Ax-b) < tol
     max_iter: the maximum number of gradient descent steps to take.
     cond_upperbound: sets the upper bound on the matrix A's condition number.
         if cond_upperbound <= 0, the original A matrix will be used
         if cond_upperbound > 0, singluar vectors will be dropped from A until K(A) <= cond_upperbound.
     proj_flag: whether or not to project the solution to positive values or not.
    OUTPUTS
     x: the solution x(omega)
     err_list: a list of the error computed for each CG iteration.

    if you have time to implement a more general optimization algorithm
    https://www.sciencedirect.com/science/article/pii/S0895717705005297
    """
    # Check the dimension of inputs
    [Ntau,Nomega] = shape(A) # Ntau rows and Nomega cols
    b = reshape( b,(-1,1) )
    [dim1, dim2] = shape(b) # Ntau rows and 1 cols
    if( dim1 != Ntau or dim2 != 1):
        print("ERROR: b needs to be a vector of dimension (Ntau,1)")    
    [dim1, dim2] = shape(C) # Ntau rows and Ntau cols
    if( dim1 != Ntau or dim2 != Ntau):
        print("ERROR: C needs to be a covariance matrix of dimension (Ntau, Ntau)")
        
    # pre-condition the inversion to a set condition number #
    if(cond_upperbound > 0):
        print("Pre-conditioning matrix to cond(A)=", cond_upperbound)
        U,S,Vh = svd(A); V=Vh.T # svd produce A = U@diag(S)@Vh, thus we have to transpose Vh to get V 
        for i in range(Ntau):
            if cond_upperbound > S[0]/S[i] :
                rank=i+1
        A = U[:,0:rank] @ diag(S[0:rank]) @ V[:,0:rank].T 

    # precompute quantities for computational speed
    Cinv = inv(C)
    tmp0 = A.T @ Cinv @ A
    tmp1 = A.T @ Cinv @ b 

    # initialize solution and residual vector
    x = ones((Nomega,1))
    r= tmp0 @ x - tmp1
    err_list = [norm(r)];
    i=0
    while( i < max_iter and norm(r)**2 > tol ):
        z = (r.T@r) / (r.T@tmp0@r) # this produces a 1x1 matrix
        x -= z*r 
        r = tmp0 @ x - tmp1 
        err_list.append(norm(r))
        i+=1
        # positivity projection step
        if (proj_flag): 
            for j, val in enumerate(x):
                if(val < 0.0):
                    x[j] = 1e-8

    chi_sq = (A@x-b).T@Cinv@(A@x-b)
    if ( chi_sq < tol):
        print("SD converged iteration: ",i,  chi_sq )
    else:
        print("SD failed to converge, ended with ||Ax-b||^@_C=", chi_sq )
        
    return x, err_list