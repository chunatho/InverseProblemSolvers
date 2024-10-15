from numba import njit
from numpy import log10, abs
from numpy import linspace, shape, reshape, zeros, ones, diag
from numpy.linalg import norm, svd

@njit
def TIK_solve(A, b, C, cond_upperbound = 1e8, lam = -1, Nlam=1000, spacing='log', proj_flag=0, max_iter=1000, errtol=1e-8):
    """
    # Tikhonov Least Squares via SVD defined by the objective function
    # minimize ||Ax-b||^2_C, + lam*||x||^2_2
    INPUT
     A: The matrix to be inverted
     b: The data
     C: the covariance matrix of the data
     cond_upperbound: Cut off for maximum condition number of matrix. 
       default (1e8) - single precision
     lam: the regularization parameter, 
       default (-1) - Use Generalized Cross Validation (GCV) to set lam
       https://pages.stat.wisc.edu/~wahba/ftp1/oldie/golub.heath.wahba.pdf
       https://en.wikipedia.org/wiki/Ridge_regression
       OLS (0) - Remove Tikhonov Regularization  
     Nlam: number of lambda points to search over in GVC approach
     spacing: 'log' or 'linear' flag to indicate whether you ought to search lambda with linear or logarithmic spacing. 
     proj_flag: enforce positivity (algorithm 1) 
       default (0) - do not use projection step.
       project (1) - enforce that the solution must be positive.
       https://www.math.kent.edu/~reichel/publications/modmeth.pdf
     max_iter: maximum number of fixed point iterations to make in the projected tikhonov loop. 
     errtol: Check error convergence for projected Tikhonov algorithm. 
       default (1e-8) - single precision
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
        
    # determine rank of matrix based on condition number
    U,S,Vh = svd(A); V=Vh.T # svd produce A = U@diag(S)@Vh, thus we have to transpose Vh to get V 
    for i in range(Ntau):
        if cond_upperbound > S[0]/S[i] :
            rank=i+1

    B=U.T@b
    # Compute the GCV using Wahba method for setting regularizer lam
    if (lam < 0): 
        # initialize the lam grid
        if (spacing =='log'):
            lams = 10**linspace(log10(1e-8), log10(1e4), Nlam) # logarithmic spacing
        elif (spacing =='linear'):
            lams = linspace(1e-8,1e4,Nlam) # linear spacing
        else:
            print("ERROR: argument spacing must be either 'log' or 'linear'.")
            
        GCV=zeros((Nlam,1))
        for j in range(Nlam):
            tmp = zeros((3,1));
            for i in range(rank):
                tmp[0]+=lams[j]**2/(S[i]**2+lams[j])**2*B[i]**2
                tmp[2]+=lams[j]/(S[i]**2+lams[j])
            for i in range(rank,Ntau):
                tmp[1]+=B[i]**2
            GCV[j]=(tmp[0]+tmp[1])/tmp[2]**2
            if (j == 0):
                lower=GCV[j]
                GCV_index=0
            if (GCV[j] < lower):
                lower=GCV[j]
                GCV_index=j
        lam=lams[GCV_index]  
    # alternative choice for lam parameter.
    #if(lam < 0 ): 
    #    lam = S[0]*S[rank-1] 
    
    # express x as a windowed sum of the column space basis vectors,
    # https://math.stackexchange.com/questions/2065192
    x=zeros((Nomega,))
    for i in range(rank):
        x += B[i] * ( S[i]   / ( lam +  S[i]**2) ) * V[:,i]

    
    if(proj_flag):  
        # based on algorithm found here https://www.math.kent.edu/~reichel/publications/modmeth.pdf
        # Bai, Zhong-Zhi, et al. "Modulus-based iterative methods for constrained Tikhonov regularization."
        # Journal of Computational and Applied Mathematics 319 (2017): 1-13.
        print("TIK: You are constraining TIK solution to be positive")
        V=V[:,0:rank]; U=U[:,0:rank]; S = S[0:rank]
        
        # Pre compute quantities for computational savings
        B=U.T@b # project b vector into column space
        vec=zeros((Nomega,))
        for i in range(rank):
            vec += B[i] * ( S[i]   / ( lam +  S[i]**2) ) * V[:,i]            
        M = V @ diag( (lam - S**2)/( lam + S**2) ) @ V.T 
        
        # fixed point iteration 
        y = x/2.
        i=0;
        while( i < max_iter and norm(A@x-b) > errtol ):
            y = M@y + vec
            x = abs(y) + y
            i+=1
        
        if ( norm(A@x-b)<errtol):
            print("TIK converged on iteration: ",i)
        else:
            print("TIK failed to converge, ended with ||Ax-b||=", norm(A@x-b))
            
    return x