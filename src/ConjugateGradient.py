from numba import njit
from numpy import fmax
from numpy import shape, reshape, ones, diag, copy
from numpy.linalg import norm, svd, inv

@njit
def CG_solve( A, b, C, tol=1e-8, cond_upperbound = -1):
    """
    This methods solves the minimizes ||Ax-b||^2_C using the Conjugate Gradient (CG) method.
    It uses the CG method for matrices that are not real and symmetric - cite painlessCGM
    The simpler real and symmetric version is not implemented here.
    INPUTS
     A: is the inversion kernel (shaped [Ntau,Nomega])
     b: the data (shaped [Ntau,1])
     C: the covariance matrix (shaped [Ntau,Ntau])
     tol: sets the stopping condition for the algorithm as norm(Ax-b) < tol
     cond_upperbound: sets the upper bound on the matrix A's condition number.
         if cond_upperbound <= 0, the original A matrix will be used
         if cond_upperbound > 0, singluar vectors will be dropped from A until K(A) <= cond_upperbound.
    OUTPUTS
     x: the solution x(omega)
     err_list: a list of the error computed for each CG iteration.
    # At min(Nomega,Ntau) iterations sol converges to the OLS (unregularized)
    """
    [Ntau,Nomega] = shape(A) # Ntau rows and Nomega cols
    b = reshape( b,(-1,1) )
    [dim1, dim2] = shape(b) # Ntau rows and 1 cols
    if( dim1 != Ntau or dim2 != 1):
        print('ERROR: b needs to be a vector of dimension (Ntau,1)')
    [dim1, dim2] = shape(C) # Ntau rows and Ntau cols
    if( dim1 != Ntau or dim2 != Ntau):
        print('ERROR: C needs to be a covariance matrix of dimension (Ntau, Ntau)')
    max_iter=fmax(Ntau,Nomega);

    # pre-condition the inversion to a set condition number #
    if(cond_upperbound > 0):
        print("Pre-conditioning matrix to cond(A)=", cond_upperbound)
        U,S,Vh = svd(A); V=Vh.T
        for i in range(Ntau):
            if cond_upperbound > S[0]/S[i] :
                rank=i+1
        A = copy(U[:,0:rank:1] @ diag(S[0:rank:1]) @ V[:,0:rank:1].T)

    #Pre-compute matrices for computational savings
    Cinv = inv(C)
    tmp0 = A.T @ Cinv @ A
    tmp1 = A.T @ Cinv @ b

    # initial guess for solution is all ones
    x = ones((Nomega,1))
    r = tmp1 - tmp0 @ x
    p = tmp1 - tmp0 @ x

    # iterate through Krylov Space
    err_list=[];
    i=0;
    while( i < max_iter and norm(r)**2 > tol ):
        alpha =  norm(r)**2 / ( r.T @ tmp0 @ p );
        x +=alpha*p;
        r -=alpha*tmp0 @ p;
        beta =-1*(r.T@tmp0@p) / (p.T@tmp0@p) ;
        p    = r+beta*p;
        err_list.append(norm(r))
        i+=1

    chi_sq = (A@x-b).T @ Cinv @ (A@x-b)
    if ( chi_sq < tol):
        print('CG converged iteration: ',i ,  chi_sq )
    else:
        print('CG failed to converge, ended with ||Ax-b||^2_C=', chi_sq )

    return x, err_list
