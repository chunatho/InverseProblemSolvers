from numba import njit

from numpy.linalg import solve
from numpy import log10, abs
from numpy import shape, reshape, linspace, zeros, ones, sum

@njit # just in time compile to speed up program
def BGM_solve_test(A, b, C, omegas, lam_input=-1, Nlam=100, spacing='log', delta=1e-8):

    # This methods solves the problem Ax=b using the Backus-Gilbert Method
    # The original BGM is laid out in Hansen et al. https://doi.org/10.1103/PhysRevD.99.094508
    #  A: is the inversion kernel (shaped [Ntau,Nomega])
    #  b: the data (shaped [Ntau,1])
    #  C: the covariance matrix (shaped [Ntau,Ntau])
    #  lam_input: the regularization parameter.
    #     if lam_input < 0, you will search over possible lambda to find the value that optimizes
    #     the Backus-Gilbert cost function best.
    #  NLam: the number of lambda values to search over
    #  spacing: ('log' or 'linear') flag to indicate whether you ought to search lambda with linear or logarithmic spacing. 
    #  delta: tolerance for normalization constraint, i.e., |q.T@R - 1| < delta
    # OUTPUTS
    #  x: the solution x(omega)
    #  q_list: list of the basis coefficients at each omega
    #  obj_list: list of the estimated objective function for each lam at each omega

    # ---------------------------------- #
    # Check to make sure inputs are fine #
    # ---------------------------------- #
    [Ntau,Nomega] = shape(A) # Ntau rows and Nomega cols
    b = reshape( b,(-1,1) )
    [dim1, dim2] = shape(b) # Ntau rows and 1 cols
    if( dim1 != Ntau or dim2 != 1):
        print('ERROR: b needs to be a vector of dimension (Ntau,1)')
    [dim1, dim2] = shape(C) # Ntau rows and Ntau cols
    if( dim1 != Ntau or dim2 != Ntau):
        print('ERROR: C needs to be a covariance matrix of dimension (Ntau, Ntau)')
    dw = omegas[1]-omegas[0]
    x=zeros((Nomega,))
    R=dw*A@ones((Nomega,1)) #Row Sum

    # ------------------------------- #
    # Compute Backus-Gilbert Estimate #
    # ------------------------------- #
    q_list=[];
    obj_list=[];
    for i in range(Nomega):
        # Compute the Variance Matrix for cost function
        W=zeros((Ntau,Ntau))
        for j in range(Ntau):
            for k in range(Ntau):
                for l in range(Nomega):
                    W[j,k]+=(omegas[l]-omegas[i])**2 * A[j,l]*A[k,l]; # eqn 17
                W[j,k] *= dw

        #  to find which lam optimizes the cost function.
        if(lam_input < 0):
            # initialize the lam grid
            if (spacing =='log'):
                lams = 10**linspace(log10(1e-5), log10(.999), Nlam) # logarithmic spacing
            elif (spacing =='linear'):
                lams = linspace(1e-3,.999,Nlam) # linear spacing
            else:
                print('ERROR: argument spacing must be either log or linear.')

            # estimate objfxn over the lam grid
            objfxn=[];
            for j in range(Nlam):
                # Naive way to compute solution
                #q=( inv(W+lams[j]*C) @R ) / (R.T @ inv(W+lams[j]*C) @ R); # eqn 22
                # numerical recipes approach
                y = solve( (1-lams[j])*W + lams[j]*C, R)
                q = y / (R.T @ y )[0,0]; # eqn 22
                val = (1.-lams[j])*sum( (omegas - omegas[i])**2 * (q.T @ A)**2) + lams[j]*q.T @ C @ q
                objfxn.append(val[0,0])
            obj_list.append(objfxn) #  store objfxn list

            # search over lam grid for objfxn maximum
            maxval = objfxn[0]
            for index in range( len(objfxn) ):
                if ( objfxn[index] >= maxval):
                    maxval = objfxn[index]
                    maxindex = index

            # compute y for lam where objfxn is optimal
            y = solve( (1-lams[maxindex])*W + lams[maxindex]*C, R)

        # use input lam value
        else:
            y = solve( (1-lam_input)*W + lam_input*C, R)

        q = y / (R.T @ y )[0,0];
        # Check that q satisfies the relevant constraints
        if ( abs( (q.T@R)[0,0]-1 ) > delta ):
            print('Something is going wrong, q.T @ R != 1')
        q_list.append(q)
        x[i]=(dw*b.T @ q)[0,0];
    return x, q_list, obj_list
