import numpy as np; pi=np.pi
from numpy.linalg import norm, svd
from scipy.special import erf

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

# ----------------------- #
# ---- Skew Gaussian ---- #
# ----------------------- #
#Nomega = 100;
#omegas = np.linspace(3.5/Nomega, 4.5, Nomega-1, endpoint=True)  # units GeV
#dw = omegas[1] - omegas[0] # units GeV
#Ntau = 75;
#taus = np.linspace(0, 1.5, Ntau);  # units GeV
#dtau = taus[1] - taus[0];
#print(Nomega, Ntau)
#print('dw', dw, 'dtau', dtau)

Nomega = 150;
omegas = np.linspace(4.0/Nomega, 4.0, Nomega-1, endpoint=True)  # units GeV
dw = omegas[1] - omegas[0] # units GeV
Ntau = 30;
taus = np.linspace(0, 5.0, Ntau);  # units GeV
dtau = taus[1] - taus[0];
print(Nomega, Ntau)
print('dw', dw, 'dtau', dtau)

c1 = 1 #0.62
z1 = 1.25 # 0.74
sigma1 = 0.5;
skew = -5 # skweness + -> tail to higher omega
t = (omegas - z1) / sigma1
pdf = np.exp(-0.5 * t**2) / np.sqrt(2*np.pi)
cdf = 0.5 * (1 + erf(skew * t / np.sqrt(2)))
x = c1 * 2 / sigma1 * pdf * cdf

print('x', x)
mu = (np.sum(x)/len(x))*np.ones_like(x)

# ------------------------------- #
# ---- Create Synthetic Data ---- #
# ------------------------------- #
noise_level=0.001
noiselevel='0p001'
if (1 == 0): #generate data
    Nsamples=100
    b_samples = np.zeros((Nsamples,Ntau))
    EE, TT = np.meshgrid(omegas, taus)
    A = TT*EE
    for i in range(len(taus)):
        for j in range(len(omegas)):
            A[i,j]= np.exp(-A[i,j])

    b_clean = A @ x
    stdev = noise_level #* b_clean
    for i in range(Nsamples):
        noise = np.random.normal(np.zeros((Ntau,)), stdev)
        b_samples[i] = b_clean + noise

    b_avg = np.average(b_samples, axis=0)
    C_iid = np.diag(np.var(b_samples, axis=0, ddof=1)/Nsamples)
    #C_full = (b_samples-b_avg).T @ (b_samples-b_avg) / Nsamples
    np.savez(f'data/skewgaussian_noise{noiselevel}_skew{skew}.npz', A=A, x=x, b_avg=b_avg, Cov=C_iid, mu=mu, taus=taus, omegas=omegas) # .npy extension is added if not given

print('loading data saved to: skewgaussian_noise{noiselevel}_skew{skew}.npz')
npz = np.load(f'data/skewgaussian_noise{noiselevel}_skew{skew}.npz', allow_pickle=True)
x=npz['x']; mu=npz['mu']; A=npz['A']; b_avg=npz['b_avg']; Cov=npz['Cov']; taus=npz['taus']; omegas=npz['omegas'];
Ntau=len(taus); Nomega=len(omegas)
b_clean = A@x
b_err = np.sqrt(np.diag(Cov))
grad_scale = 1e-3/np.linalg.norm(b_err/b_avg[0])
normalization = b_avg[0]
print('relative error in b', np.linalg.norm(b_err) / np.linalg.norm(b_avg) )
print('A condition number', np.linalg.cond(A))

# asses the data
res = np.reshape( b_avg - b_clean.flatten(), (-1,1))
b0_chi2 = (res.T@np.linalg.inv(Cov)@res)[0,0]/Ntau
print('chi-sq of true data', b0_chi2)

# create normalized quantitites for routine
b_input = np.reshape( b_avg/normalization, (-1,1))
C_iid = np.diag(np.diag(Cov)/(normalization**2))
m = mu/normalization

print('x sum',np.sum(x), 'b_avg[0]', normalization, (A@x)[0], 'expected normalization', np.sum(x) / b_avg[0] )
print('x sum / normalized', np.sum(x)/normalization)
print('m sum', np.sum(m))

# ---------------------------- #
# Do Data Inversion with noise #
# ---------------------------- #
# stochastic optimization using set of Gaussians and coefficients are set with regularized optimization
Nkernels_min=1
Nkernels_max=5
tmp = RSOMpy.RSOM_solve(A, b_input, C_iid, m, omegas, taus,
                        step_size=2e-2, beta_min=1e2, beta_max=1e4, Nbeta=110,
                        Npt=4000, Nmh=50, burn_in=200, Nsamples=20000, # MH solution collection
                        genetic_flag=1, p_reset=0.001, # whether to use genetic reset
                        lambda_min=1e-6, lambda_max=0.5,
                        Nkernels_min=Nkernels_min, Nkernels_max=Nkernels_max, Nkernels_step=1,
                        grad_scale=grad_scale)

x_RSOM, x_RSOM_var, RSOM_proposedsolutions, RSOM_chi2, RSOM_choice = tmp
print('shape RSOM', np.shape(x_RSOM), np.shape(x_RSOM_var))


alpha_min=1e-1
alpha_max=1e5
Nalpha=31
# Entropic Methods
print('data inversions with noise')
tmp = MEMdNpy.MEMdN_solve(A, b_input, C_iid, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                          rtol=1e-6, atol =1e-6)
x_MEMdN, x_MEMdN_var, proposed_solutions_MEMdN, P_MEMdN, acceptance_arr_MEMdN = tmp
print('dual', P_MEMdN)
print('dual', acceptance_arr_MEMdN)

tmp = MEMBpy.MEMBryan_solve(A, b_input, C_iid, m, alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha)
x_MEMB, x_MEMB_var, proposed_solutions_MEMB, P_MEMB, acceptance_arr_MEMB = tmp
print('primal', P_MEMB)
print('primal', acceptance_arr_MEMB)

# Entropic Regularization with Chi-2-kink algorithm for alpha selection
alpha_min=1e-1
alpha_max=1e8
Nalpha=31
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


# Backus-Gilbert Methods
#x_BG, qlist_BG, obj_list_BG = BGpy.BG_solve(A, b, C, omegas )
#x_GBG, glist_GBG, obj_list_GBG, smearing_list_GBG = GBGpy.GaussianBG_solve(A, b, C, omegas, taus, sigma=1)

val = normalization
plt.figure()
plt.plot(omegas, x, label='solution', c='black', marker='o', markevery=21)
plt.plot(omegas, mu, label='prior',  c='black', marker='x', markevery=13, alpha=0.5)
#plt.plot(omegas, x_BG, label='Backus-Gilbert' )
#plt.plot(omegas, x_GBG, label='Gaussian Backus-Gilbert' )
plt.plot(omegas, val*x_RSOM, label='RSOM', c='darkorange', marker='h', markevery=15 )
plt.plot(omegas, val*(x_RSOM+2*np.sqrt(x_RSOM_var)), c='darkorange', ls='--', label=r'error', alpha=.5)
plt.plot(omegas, val*(x_RSOM-2*np.sqrt(x_RSOM_var)), c='darkorange', ls='--', alpha=.5)

plt.plot(omegas, val*x_C2KB, label=r'$\chi^2$k-B', c='blue', marker='s', markevery=17, ms=5)
plt.plot(omegas, val*(x_C2KB+2*np.sqrt(x_C2KB_var)), c='blue', ls='--', label=r'error', alpha=.5)
plt.plot(omegas, val*(x_C2KB-2*np.sqrt(x_C2KB_var)), c='blue', ls='--', alpha=.5)

plt.plot(omegas, val*x_C2KdN, label=r'$\chi^2$k-dN', c='red', marker='^', markevery=19, ms=5)
plt.plot(omegas, val*(x_C2KdN+2*np.sqrt(x_C2KdN_var)), c='red', ls='--', label=r'error', alpha=.5)
plt.plot(omegas, val*(x_C2KdN-2*np.sqrt(x_C2KdN_var)), c='red', ls='--', alpha=.5)
plt.xlabel(r'$\omega$ [unitless]', fontsize=16)
plt.xlim([omegas[0], omegas[-1]])
plt.ylim([0, 1.2*x.max()])
plt.legend(fontsize=14, loc='upper right')
plt.tick_params(axis='both', labelsize=16)
print(f'imgs/skewgaussian_reconstructions_noise{noiselevel}_prior-flat_skew{skew}.pdf')
plt.savefig(f'imgs/skewgaussian_reconstructions_noise{noiselevel}_prior-flat_skew{skew}.pdf', bbox_inches='tight', format='pdf')

# Plot data
plt.figure()
plt.title('Transformed solutions compared to data', fontsize=16)
plt.errorbar(taus, b_avg, b_err, color='black', label='avg +/- err')
plt.plot(taus, A@x, color='black', label='$b^0$'+ r', $\tilde{\chi}^2 = $'+f'{b0_chi2:.3f}', alpha=0.5)
#plt.plot(taus, A@x, color='black', label='$b^0$')
#plt.fill_between(taus, b_avg - b_err, b_avg + b_err, color='black', label='avg +/- err', alpha=0.5)

sols = [x_RSOM, x_C2KB, x_C2KdN]
colors=['darkorange', 'blue', 'red']
markers=['h', 's', '^']
labels=['RSOM', r'$\chi^2$k-B', r'$\chi^2$k-dN']
for i, x_sol in enumerate(sols):
    x_sol  = np.reshape(x_sol, (-1, 1))
    res = A@x_sol - b_input
    chi2 = (res.T@np.diag(1./np.diag(C_iid))@res/Ntau)
    print(labels[i], 'chi2', chi2)
    chi2 = (res.T@np.diag(1./np.diag(C_iid))@res/Ntau)[0,0]
    plt.plot(taus, val*A@x_sol, color=colors[i], marker=markers[i], markevery=3+i, ms=10, label=labels[i] + r', $\tilde{\chi}^2 = $'+f'{chi2:.3f}')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\tau$ [unitless]', fontsize=16)
plt.yscale('log')
plt.legend(fontsize=15)
print(f'saved figute to: imgs/skewgaussian_syntheticdata_noise{noiselevel}_prior-flat_skew{skew}.pdf')
plt.savefig(f'imgs/skewgaussian_syntheticdata_noise{noiselevel}_prior-flat_skew{skew}.pdf', bbox_inches='tight', format='pdf')


# RSOM diagnostics
plt.figure()
plt.plot(omegas[3:], x[3:], color='black', label='solution', marker='.', markevery=4)
for i, sol in enumerate(RSOM_proposedsolutions):
    if i == RSOM_choice:
        plt.plot(omegas, val*x_RSOM, label=f'BEST, $N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', color='darkorange', marker='h', markevery=4)
    else:
        plt.plot(omegas, val*sol, label=f'$N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', alpha=0.5)

plt.xlim([omegas[0], omegas[-1]])
plt.xlabel(r'$\omega$ [unitless]', fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.tick_params(axis='both', labelsize=16)
#plt.tick_params(axis='both', which='major', labelsize=10)
#plt.tick_params(axis='both', which='minor', labelsize=8)
plt.title('RSOM solutions across number of kernels', fontsize=16)
print(f'saved figure: imgs/skewgaussian_dictionarylearning_sols_noise{noiselevel}_skew{skew}.pdf')
plt.savefig(f'imgs/skewgaussian_dictionarylearning_sols_noise{noiselevel}_skew{skew}.pdf', bbox_inches='tight', format='pdf')
#plt.savefig('imgs/skewgaussian_dictionarylearning_sols_noise'+noiselevel+'.pdf', bbox_inches='tight', format='pdf')
