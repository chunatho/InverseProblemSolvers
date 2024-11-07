from numba import njit

from numpy.linalg import solve
from numpy import log10, exp, abs, sqrt, pi
from numpy import shape, reshape, linspace, zeros, ones, sum
from math import erf

@njit # just in time compile to speed up program
def GaussianBGM_solve(A, b, C, omegas, taus, sigma=.01, eps=1e-6, lam_input=-1, Nlam=100, spacing='log', delta=1e-8, periodic=-1):
    """ 
    This methods solves the problem Ax=b using the Gaussian Backus-Gilbert Method (BGM) 
    The Gaussian BGM is laid out in Hansen et al. https://doi.org/10.1103/PhysRevD.99.094508
    INPUTS
     A: the transform kernel to be inverted (shaped [Ntau,Nomega])
     b: the data G(tau) (shaped [Ntau,1])
     C: the covariance matrix of b (shaped [Ntau,Ntau])
     omegas: the linspace of omega points (shape [Nomega,])
     taus: the linspace of tau points
     lam_input: the regularization parameter.
        if lam_input < 0, you will search over possible lambda to find the value that optimizes
        the Backus-Gilbert cost function best.
     NLam: the number of lambda values to search over
     spacing: 'log' or 'linear' flag to indicate whether you ought to search lambda with linear or logarithmic spacing. 
     delta: tolerance for normalization constraint, i.e., |q.T@R - 1| < delta 
     periodic: flag to specify whether to use analytic formulas or numeric computation
        case -1, you will compute the formulas numerically (default)
        case 0, you will use analytic formula for non-periodic Laplace (not well tested)
        case 1, you will use analytic formula for periodic Laplace (not well tested)
    OUTPUTS
     x: the solution x(omega)
     g_list: list of the basis coefficients at each omega
     obj_list: list of the estimated objective function for each lam at each omega
     smearing_list: list of the specified smearing functions
    """
    # Check the dimension of inputs
    [Ntau,Nomega] = shape(A) 
    b = reshape( b,(-1,1) )
    [dim1, dim2] = shape(b) # Ntau rows and 1 cols
    if( dim1 != Ntau or dim2 != 1):
        print("ERROR: b needs to be a vector of dimension (Ntau,1)")    
    [dim1, dim2] = shape(C) # Ntau rows and Ntau cols
    if( dim1 != Ntau or dim2 != Ntau):
        print("ERROR: C needs to be a covariance matrix of dimension (Ntau, Ntau)")
    
    x=zeros((Nomega,))
    T = taus[-1]
    E0 = omegas[0]
    dw = omegas[1]-omegas[0]
    if(periodic==0): # use non-periodic formula
        R = reshape(1./(taus+eps), (Ntau,1))
    elif(periodic==1): # use periodic formula
        R = reshape(1./(taus+eps) + 1./(T - taus + eps), (Ntau,1))
    elif(periodic==-1): # compute numerically
        R = dw*A@ones((Nomega,1), dtype='float64') #Row Sum of Kernel

    # ---------------------------------------- #
    # Compute Gaussian Backus-Gilbert Estimate #
    # ---------------------------------------- #
    g_list=[]; 
    obj_list=[];
    smearing_list=[];
    for i in range(Nomega):
        # Compute the Variance Matrix for Gaussian BGM cost function
        W=zeros((Ntau,Ntau))
        for j, t in enumerate(taus ):
            for k, r in enumerate(taus ):
                if(periodic==0): # use non=periodic formula
                    W[j,k] = exp( -(r+t+eps)*E0 )/(r+t+eps)
                elif(periodic==1): # use periodic formula
                    W[j,k] = exp( -(r+t+eps)*E0 )/(r+t+eps) + exp( -(T-r+t+eps)*E0 )/(T-r+t+eps) \
                                + exp( -(T+r-t+eps)*E0 )/(T+r-t+eps) + exp( -(2.*T-r-t+eps)*E0 )/(2.*T-r-t+eps)
                elif(periodic==-1): # compute numerically
                    for l in range(Nomega):
                        W[j,k]+= dw*A[j,l]*A[k,l]        

        # Compute the normalized Gaussians to be used as smearing function
        Z = 1./2. * ( 1. + erf(omegas[i] / (sqrt(2.)*sigma)) ) # eq A2
        smearing = 1./( sqrt(2.*pi)*sigma*Z ) * exp( -(omegas - omegas[i])**2 / (2.*sigma**2) ); # eq A1
        smearing_list.append(smearing)

        # compute f from eqn ???
        if(periodic==0): # use non=periodic formula
            f=zeros((Ntau,1))
            for j in range( Ntau ):
                f[j] = Nfxn(taus[j], omegas[i], Z, sigma, eps) * Ffxn(taus[j], omegas[i], omegas[0], sigma, eps)  
        elif(periodic==1): # use periodic formula
            f=zeros((Ntau,1))
            for j in range( Ntau ):
                f[j] = Nfxn(taus[j], omegas[i], Z, sigma, eps) * Ffxn(taus[j], omegas[i], omegas[0], sigma, eps) \
                       + Nfxn(taus[-1]-taus[j], omegas[i], Z, sigma, eps) * Ffxn(taus[-1]-taus[j], omegas[i], omegas[0], sigma, eps) 
        elif(periodic==-1): # compute numerically
            f = dw*A@reshape(smearing, (-1,1)) # eq 30
            
        if(lam_input < 0):
            # initialize the lam grid
            if (spacing =='log'):
                lams = 10**linspace(log10(1e-5), log10(.999), Nlam) # logarithmic spacing
            elif (spacing =='linear'):
                lams = linspace(1e-5,.999,Nlam) # linear spacing
            else:
                print("ERROR: argument spacing must be either 'log' or 'linear'.")
                
            # estimate objfxn over the lam grid            
            objfxn=[];
            for lam in lams:
                z = solve( (1.-lam)*W + lam/(b[0]**2)*C, (1.-lam)*f)
                y = solve( (1.-lam)*W + lam/(b[0]**2)*C, R)
                g = z + y*( 1. - R.T@z)[0,0] / (R.T @ y )[0,0]; # eqn 22
                val = (1.-lam)*sum( (smearing - g.T @ A)**2 ) + lam/(b[0]**2)*g.T @ C @ g
                objfxn.append(val[0,0])
            obj_list.append(objfxn) #  store objfxn list 
            
            # search over lam grid for objfxn maximum
            maxval = objfxn[0]
            for index in range( len(objfxn) ):
                if ( objfxn[index] >= maxval):
                    maxval = objfxn[index]
                    maxindex = index
                    
            # compute z,y for lam where objfxn is optimal
            z = solve( (1-lams[maxindex])*W + lams[maxindex]*C / b[0]**2, (1-lams[maxindex])*f )
            y = solve( (1-lams[maxindex])*W + lams[maxindex]*C / b[0]**2, R)

        else:
            # compute z,y for given lam
            z = solve( (1-lam_input)*W + lam_input*C / b[0]**2, (1-lam_input)*f)
            y = solve( (1-lam_input)*W + lam_input*C / b[0]**2, R)
            
        g = z + y*( 1 - R.T@z)[0,0] / (R.T @ y )[0,0]; # eqn 22
        # Check that q satisfies the relevant constraints
        if ( abs( (g.T@R)[0,0]-1 ) > delta ):
            print("Something is going wrong, g.T @ R != 1")
        
        g_list.append(g)
        x[i]=(dw*b.T @ g)[0,0];
    return x, g_list, obj_list, smearing_list

# Helper Functions
@njit
def Nfxn(t, omega, Z, sigma, eps):
    return 1./(2.*Z) * exp( (eps-t) * ((eps-t)*sigma**2.+2.*omega) / 2. ) # eq A8
@njit
def Ffxn(t, omega, omega0, sigma, eps):
    return 1. + erf( ((eps-t)*sigma**2 + omega - omega0 )/ (sqrt(2.)*sigma) )  # eq A9
