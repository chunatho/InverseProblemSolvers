import numpy as np; pi=np.pi
from numpy.linalg import norm, svd
import matplotlib.pyplot as plt
from matplotlib import cm
colormap = cm.viridis  # You can change to 'plasma', 'jet', etc.
import time

# Call InverseProblemSolvers library
#import src.BackusGilbert as BGpy
#import src.GaussianBackusGilbert as GBGpy
import src.RSOM_langevin as RSOMpy
import src.MEMBryan as MEMBpy
import src.MEMdualNewton as MEMdNpy

import src.C2KBryan as C2KBpy
import src.C2KdualNewton as C2KdNpy

# ------------------------- #
# ---- Double Gaussian ---- #
# ------------------------- #
# This spectrum is text case 1 from Goulko, Olga, et al.
# "Numerical analytic continuation: Answers to well-posed questions."
# Physical Review B 95.1 (2017): 014102.
Nomega = 150;
omegas = np.linspace(4.0/Nomega, 4.0, Nomega, endpoint=True)
dw = omegas[1] - omegas[0] # unitless
Ntau = 30;
taus = np.linspace(0, 5, Ntau);  # unitless
dtau = taus[1] - taus[0];
print(Nomega, Ntau)
print('dw', dw, 'dtau', dtau)

c1=0.62; z1=0.74; sigma1=0.12;
c2=0.41; z2=2.93; sigma2=0.064;
x = c1/sigma1*np.exp(-(omegas-z1)**2/2/sigma1**2 )
x += c2/sigma2*np.exp(-(omegas-z2)**2/2/sigma2**2 )
mu = np.sum(x)*(np.ones_like(x)/len(x))
# ------------------------------- #
# ---- Create Synthetic Data ---- #
# ------------------------------- #
noise_level=0.01
noiselevel='0p01'
if (0 == 1): #generate data
    Nsamples=100
    b_samples = np.zeros((Nsamples,Ntau))
    EE, TT = np.meshgrid(omegas, taus)
    A = TT*EE
    for i in range(len(taus)):
        for j in range(len(omegas)):
            A[i,j]= np.exp(-A[i,j])

    b_clean = A @ x
    stdev = noise_level * b_clean
    for i in range(Nsamples):
        noise = np.random.normal(np.zeros((Ntau,)), stdev)
        b_samples[i] = b_clean + noise

    b_avg = np.average(b_samples, axis=0)
    C_iid = np.diag(np.var(b_samples, axis=0, ddof=1)/Nsamples)
    #C_full = (b_samples-b_avg).T @ (b_samples-b_avg) / Nsamples
    np.savez('data/double-gaussian_noise'+noiselevel+'.npz', A=A, x=x, b_avg=b_avg, Cov=C_iid, mu=mu, taus=taus, omegas=omegas) # .npy extension is added if not given

# Load Ax=b problem data
print('loading data saved to: doublegaussian_noise'+noiselevel+'.npz')
npz = np.load('data/double-gaussian_noise'+noiselevel+'.npz', allow_pickle=True)
x=npz['x']; mu=npz['mu']; A=npz['A']; b_avg=npz['b_avg']; Cov=npz['Cov']; taus=npz['taus']; omegas=npz['omegas'];
Ntau=len(taus); Nomega=len(omegas)
b_clean = A@x
b_err = np.sqrt(np.diag(Cov))
normalization = b_avg[0]

# asses the difficulty of the Ax=b problem
print('A condition number', np.linalg.cond(A))
res = np.reshape( b_avg - b_clean.flatten(), (-1,1))
b0_chi2 = (res.T@np.linalg.inv(Cov)@res)[0,0]/Ntau
print('chi-sq of true data', b0_chi2)
print('relative error in b', np.linalg.norm(b_err) / np.linalg.norm(b_avg) )

# create normalized quantitites for solvers
grad_scale = 1e-3/np.linalg.norm(b_err/b_avg[0]) # what counts as a small gradient?
b_input = np.reshape( b_avg/normalization, (-1,1))
C_iid = np.diag(np.diag(Cov)/(normalization**2))
m = mu/normalization

# Check normalizations
print('x sum',np.sum(x), 'b_avg[0]', normalization, (A@x)[0], 'expected normalization', np.sum(x) / b_avg[0] )
print('x sum / normalized', np.sum(x)/normalization)
print('m sum', np.sum(m))
print('expected grad scale:', grad_scale)

# ---------------------------- #
# Do Data Inversion with noise #
# ---------------------------- #
Nalpha=51
alpha_min=1e-2
alpha_max=1e13
# Entropic Methods
print('data inversions with noise')
tmp = MEMdNpy.MEMdN_solve(A, b_input, C_iid, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                          rtol=1e-6, atol=1e-6)
x_MEMdN, x_MEMdN_var, proposed_solutions_MEMdN, P_MEMdN, acceptance_arr_MEMdN = tmp
print('dual', P_MEMdN)
print('dual', acceptance_arr_MEMdN)

tmp = MEMBpy.MEMBryan_solve(A, b_input, C_iid, m, alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha)
x_MEMB, x_MEMB_var, proposed_solutions_MEMB, P_MEMB, acceptance_arr_MEMB = tmp
print('primal', P_MEMB)
print('primal', acceptance_arr_MEMB)
#print('primal', x_MEMB)

# Backus-Gilbert Methods
#x_BG, qlist_BG, obj_list_BG = BGpy.BG_solve(A, b, C, omegas )
#x_GBG, glist_GBG, obj_list_GBG, smearing_list_GBG = GBGpy.GaussianBG_solve(A, b, C, omegas, taus, sigma=1)

# Entropic Regularization with Chi-2-kink algorithm for alpha selection
C2K_min = 2.5
C2K_max = 3.0
NC2K = 10
tmp = C2KBpy.C2KBryan_solve(A, b_input, C_iid, m,
                            alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                            C2K_min=C2K_min, C2K_max=C2K_max, NC2K=NC2K)
x_C2KB, x_C2KB_var, C2KB_proposedsolutions, C2KB_chi2, C2KB_best_alphas = tmp
print('shape C2KB', np.shape(x_C2KB), np.shape(x_C2KB_var))

tmp = C2KdNpy.C2KdN_solve(A, b_input, C_iid, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                          C2K_min=C2K_min, C2K_max=C2K_max, NC2K=NC2K,
                          rtol=1e-6, atol=1e-6)
x_C2KdN, x_C2KdN_var, C2KdN_proposedsolutions, C2KdN_chi2, C2KdN_best_alphas = tmp
print('shape C2KdN', np.shape(x_C2KdN), np.shape(x_C2KdN_var))

# stochastic optimization using set of Gaussians and coefficients are set with regularized optimization
Nkernels_min=1
Nkernels_max=4
tmp = RSOMpy.RSOM_solve(A, b_input, C_iid, mu, omegas, taus,
                        step_size=1e-1, beta_min=1e2, beta_max=1e5, Nbeta=250,
                        Npt=6000, Nmh=20, burn_in=1000, Nsamples=30000, # MH solution collection burn_in
                        genetic_flag=1, p_reset=0.01, #0.005, # whether to use genetic reset
                        lambda_min=1e-8, lambda_max=0.6,
                        Nkernels_min=Nkernels_min, Nkernels_max=Nkernels_max, Nkernels_step=1,
                        grad_scale=grad_scale)


#                        step_size=8e-5, beta_min=1e2, beta_max=1e4, Nbeta=250,
#Npt 5000 Nmh 20 burn_in 100 Nsamples 10000

x_RSOM, x_RSOM_var, RSOM_proposedsolutions, RSOM_chi2, RSOM_choice = tmp
print('shape RSOM', np.shape(x_RSOM), np.shape(x_RSOM_var))


val = normalization
# Plot solutions
plt.figure()
plt.title('Inverse problem solutions', fontsize=16)
plt.plot(omegas, x, label='solution', c='black', marker='o', markevery=35)
plt.plot(omegas, mu, label='prior',  c='black', marker='x', markevery=17, alpha=0.5)
#plt.plot(omegas, x_BG, label='Backus-Gilbert' )
#plt.plot(omegas, x_GBG, label='Gaussian Backus-Gilbert' )
plt.plot(omegas, val*x_RSOM, label='RSOM', c='darkorange', marker='h', markevery=67 )
plt.plot(omegas, val*(x_RSOM+2*np.sqrt(x_RSOM_var)), c='darkorange', ls='--', label=r'error', alpha=0.5)
plt.plot(omegas, val*(x_RSOM-2*np.sqrt(x_RSOM_var)), c='darkorange', ls='--', alpha=0.5)

#plt.plot(omegas, val*x_MEMB, label='MEM-B', c='black', marker='s', markevery=55, ms=5, alpha=0.2)
#plt.plot(omegas, val*(x_MEMB+2*np.sqrt(x_MEMB_var)), c='blue', ls='--', label=r'error', alpha=0.5)
#plt.plot(omegas, val*(x_MEMB-2*np.sqrt(x_MEMB_var)), c='blue', ls='--', alpha=0.5)

#plt.plot(omegas, val*x_MEMdN, label='MEM-dN', c='black', marker='^', markevery=42, ms=5, alpha=0.2)
#plt.plot(omegas, val*(x_MEMdN+2*np.sqrt(x_MEMdN_var)), c='red', ls='--', label=r'error', alpha=0.5)
#plt.plot(omegas, val*(x_MEMdN-2*np.sqrt(x_MEMdN_var)), c='red', ls='--', alpha=0.5)

plt.plot(omegas, val*x_C2KB, label=r'$\chi^2$k-B', c='blue', marker='s', markevery=55, ms=5)
plt.plot(omegas, val*(x_C2KB+2*np.sqrt(x_C2KB_var)), c='blue', ls='--', label=r'error', alpha=0.5)
plt.plot(omegas, val*(x_C2KB-2*np.sqrt(x_C2KB_var)), c='blue', ls='--', alpha=0.5)

plt.plot(omegas, val*x_C2KB, label=r'$\chi^2$k-dN', c='red', marker='^', markevery=42, ms=5)
plt.plot(omegas, val*(x_C2KB+2*np.sqrt(x_C2KB_var)), c='red', ls='--', label=r'error', alpha=0.5)
plt.plot(omegas, val*(x_C2KB-2*np.sqrt(x_C2KB_var)), c='red', ls='--', alpha=0.5)

plt.xlabel(r'$\omega$ [unitless]', fontsize=16)
plt.xlim([omegas[0], omegas[-10]])
plt.ylim([0, 1.2*x.max()])
plt.legend(fontsize=14, loc='upper center')
plt.tick_params(axis='both', labelsize=16)
print('imgs/doublegaussian_reconstructions_noise'+noiselevel+'_prior-flat.pdf')
plt.savefig('imgs/doublegaussian_reconstructions_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# Plot data
plt.figure()
plt.title('Transformed solutions compared to data', fontsize=16)
plt.errorbar(taus, b_avg, b_err, color='black', label='avg +/- err')
plt.plot(taus, A@x, color='black', label='$b^0$'+ r', $\tilde{\chi}^2 = $'+f'{b0_chi2:.3f}', alpha=0.5)
sols = [x_RSOM, x_MEMB, x_MEMdN]
colors=['darkorange', 'blue', 'red']
markers=['h', 's', '^']
labels=['RSOM', r'$\chi^2$k-B', r'$\chi^2$k-dN']
for i, x_sol in enumerate(sols):
    x_sol  = np.reshape(x_sol, (-1, 1))
    res = A@x_sol - b_input
    chi2 = (res.T@np.diag(1./np.diag(C_iid))@res)[0,0]/Ntau
    plt.plot(taus, normalization*A@x_sol, color=colors[i], marker=markers[i], markevery=3+i, ms=10, label=labels[i] + r', $\tilde{\chi}^2 = $'+f'{chi2:.3f}')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\tau$ [unitless]', fontsize=16)
plt.yscale('log')
plt.legend(fontsize=15)
plt.savefig('imgs/doublegaussian_syntheticdata_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')


# RSOM diagnostics
plt.figure()
plt.plot(omegas[3:], x[3:], color='black', label='solution', marker='.', markevery=4)
for i, sol in enumerate(RSOM_proposedsolutions):
    if i == RSOM_choice:
        plt.plot(omegas, val*x_RSOM, label=f'BEST, $N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', color='darkorange', marker='h', markevery=4)
    else:
        plt.plot(omegas, val*sol, label=f'$N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', alpha=0.5)

plt.xlim([omegas[0], omegas[-10]])
plt.xlabel(r'$\omega$ [unitless]', fontsize=14)
plt.legend(loc='upper left', fontsize=15)
plt.tick_params(axis='both', labelsize=16)
#plt.tick_params(axis='both', which='major', labelsize=10)
#plt.tick_params(axis='both', which='minor', labelsize=8)
plt.title('RSOM solutions across number of kernels', fontsize=16)
print('saved figure: imgs/doublegaussian_dictionarylearning_sols_noise'+noiselevel+'.pdf')
plt.savefig('imgs/doublegaussian_dictionarylearning_sols_noise'+noiselevel+'.pdf', bbox_inches='tight', format='pdf')
