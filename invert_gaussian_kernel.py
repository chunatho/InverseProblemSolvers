import numpy as np
rng = np.random.default_rng(seed=978981)
import src.SteepestDescent as SDpy
import src.ConjugateGradient as CGpy
import src.Tikhonov as TIKpy
import src.BackusGilbert as BGpy
import src.GaussianBackusGilbert as GBGpy
import src.C2KBryan as C2KBpy
import src.MEMBryan as MEMBpy
import src.RSOM_langevin as RSOMpy
dN = True
if(dN ==True):
    import src.C2KdualNewton as C2KdNpy
    import src.MEMdualNewton as MEMdNpy
import matplotlib.pyplot as plt

# Specify discretization
Nomega = 100; Ntau = 60;
omegas = np.linspace(4./Nomega, 4, Nomega)
taus = np.linspace(4./Ntau, 4, Ntau)

# Construct Signal
x = 5 + np.sin( np.pi / 2 * omegas )
normalization = np.sum(x)
x /= normalization

# Construct Gaussian Kernel
def K(taus, omegas, beta=10):
    EE, TT = np.meshgrid(omegas, taus)
    return np.exp( -beta/2.*(TT-EE)**2)
beta=10
A = K(taus, omegas, beta)
b_clean = A@x

#declare Bayesian Prior
c = 1.0 # 1.0 creates flat prior
mu = (1-c)*np.reshape(x, (-1,1)) + c*np.ones((Nomega,1))/Nomega

# Construct data so shape(b_samples) -> [Nsamples, Ntau]
noise_level = 1e-2
noiselevel='0p01'
if ( 0 == 1 ):
    Nsamples=100
    b_samples = np.tile( b_clean, (Nsamples,1))
    for i in range(Nsamples):
        noise = rng.normal(np.zeros((Ntau,)), noise_level)
        b_samples[i] += noise

    b_avg = np.average(b_samples, axis=0)
    C_iid = np.diag(np.var(b_samples, axis=0, ddof=1)/Nsamples)
    #C_full = (b_samples-b_avg).T @ (b_samples-b_avg) / Nsamples
    np.savez('data/gaussian_kernel_noise' + noiselevel + '.npz', A=A, x=x, b_avg=b_avg, Cov=C_iid, mu=mu, taus=taus, omegas=omegas) # .npy extension is added if not given

npz = np.load('data/gaussian_kernel_noise' + noiselevel + '.npz', allow_pickle=True)
A=npz['A']; x=npz['x']; b_avg=npz['b_avg']; Cov=npz['Cov']; mu=npz['mu']; taus=npz['taus']; omegas=npz['omegas'];
b_err = np.sqrt(np.diag(Cov))
C_iid = np.diag(np.diag(Cov))
Ntau=len(taus)
Nomega=len(omegas)
grad_scale = 1e-3/np.linalg.norm(b_err/b_avg[0]) # what counts as a small gradient?

res = np.reshape(b_avg - b_clean, (-1,1))
b0_chi2 = (res.T@np.linalg.inv(C_iid)@res)[0,0]/Ntau
print('reduced chi-sq of true data', b0_chi2)
print('relative error in b', np.linalg.norm(b_err)/np.linalg.norm(b_avg) )
print('A condition number', np.linalg.cond(A))
print('grad_scale', grad_scale )

# Numerical Recipes - Inverse Methods
x_CG, err_list = CGpy.CG_solve(A, b_avg, C_iid, tol=1e0, cond_upperbound=1e6) # converge when you reach chi-sq=1
x_SD, err_list = SDpy.SD_solve(A, b_avg, C_iid, tol=1e0, cond_upperbound=1e6) # converge when you reach chi-sq=1
x_BGM, qlist_BGM, obj_list_BGM = BGpy.BGM_solve(A, b_avg, C_iid, omegas) #, lam_input=0.8 )
x_GBGM, glist_GBGM, obj_list_GBGM, smearing_list_GBGM = GBGpy.GaussianBGM_solve(A, b_avg, C_iid, omegas, taus, sigma=0.22)

# Entropic Regularization with Gull's Bayesian Posterior for regularization weight selection
tmp = MEMBpy.MEMBryan_solve(A, b_avg, C_iid, mu, alpha_min=1e1, alpha_max=1e7, Nalpha=51)
x_MEMB, x_MEMB_var, MEMB_proposedsolutions, MEMB_P, MEMB_acceptance_arr = tmp

if (dN == True):
    tmp = MEMdNpy.MEMdN_solve(A, b_avg, C_iid, mu,
                              alpha_min=1e1, alpha_max=1e7, Nalpha=51,
                              rtol=1e-6, atol =1e-6)
    x_MEMdN, x_MEMdN_var, MEMdN_proposedsolutions, MEMdN_P, MEMdN_acceptance_arr = tmp

# Entropic Regularization with Chi-2-kink algorithm for alpha selection
tmp = C2KBpy.C2KBryan_solve(A, b_avg, C_iid, mu, alpha_min=1e0, alpha_max=1e10)
x_C2KB, x_C2KB_var, C2KB_proposedsolutions, C2KB_chi2, best_alphas = tmp
print('shape C2KB', np.shape(x_C2KB), np.shape(x_C2KB_var))

if (dN == True):
    tmp = C2KdNpy.C2KdN_solve(A, b_avg, C_iid, mu,
                              alpha_min=1e0, alpha_max=1e10, Nalpha=31,
                              rtol=1e-6, atol =1e-6)
    x_C2KdN, x_C2KdN_var, C2KdN_proposedsolutions, C2KdN_chi2, dN_best_alphas = tmp
    print('shape C2KdN', np.shape(x_C2KdN), np.shape(x_C2KdN_var))

# Regularized stochastic optimization using set of Gaussians and coefficients are set with regularized optimization
Nkernels_min=1
Nkernels_max=4
tmp = RSOMpy.RSOM_solve(A, b_avg, C_iid, mu, omegas, taus,
                        step_size=1e-2, beta_min=1e2, beta_max=1e4, Nbeta=120,
                        Npt=5000, Nmh=50, burn_in=150, Nsamples=10000, # MH solution collection
                        genetic_flag=1, p_reset=0.005, # whether to use genetic reset
                        Nkernels_min=Nkernels_min, Nkernels_max=Nkernels_max, Nkernels_step=1,
                        grad_scale=grad_scale)
x_RSOM, x_RSOM_var, RSOM_proposedsolutions, RSOM_chi2, RSOM_choice = tmp
print('shape RSOM', np.shape(x_RSOM), np.shape(x_RSOM_var))

# --------------------- #
#     PLOTTING CODE     #
# --------------------- #
b_avg = np.reshape(b_avg, (Ntau, 1))

# distinct colors
colors = [
    '#9467bd',  # purple
    '#2ca02c',  # green
    'darkorange',
    'blue',
    'red',
    '#8c564b',  # brown
    '#7f7f7f',  # gray
]
markers = ['^', 'o', 'h', 's', '*', 'x', '.']
labels=['BG', 'Gaussian-BG', 'RSOM', r'$\chi^2$k-B', r'$\chi^2$k-dN', 'MEM Bryan', 'Conjugate Gradient', 'steepest descent']

# Plot data
plt.figure()
plt.title('Transformed solutions compared to data', fontsize=16)
if(dN):
    sols = [x_BGM.flatten(), x_GBGM.flatten(), x_RSOM.flatten(), x_C2KB.flatten(), x_C2KdN.flatten()]
else:
    sols = [x_BGM.flatten(), x_GBGM.flatten(), x_RSOM.flatten(), x_C2KB.flatten()]
for i, x_sol in enumerate(sols):
    res = A@np.reshape(x_sol, (Nomega, 1)) - b_avg
    chi2 = (res.T@np.diag(1./np.diag(C_iid))@res/Ntau)[0,0]
    plt.plot(taus, A@x_sol+(len(sols)-i)*1e-2, color=colors[i], marker=markers[i], markevery=3+i, ms=10, label=labels[i] + r', $\tilde{\chi}^2 = $'+f'{chi2:.2f}')
plt.plot(taus, A@x, color='black', label='$b^0$'+ r', $\tilde{\chi}^2 = $'+f'{b0_chi2:.3f}', alpha=0.5)
plt.errorbar(taus, b_avg.flatten(), b_err, color='black', label='avg +/- err')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\tau$ [unitless]', fontsize=16)
plt.legend(fontsize=14, loc='lower left')
plt.savefig('imgs/gaussianInversion_syntheticdata.pdf', bbox_inches='tight', format='pdf')

# Plot Solutions
plt.figure()
plt.title('Inverse problem solutions', fontsize=16)
if (dN == True):
    vars = [0.0, 0.0, x_RSOM_var.flatten(), x_C2KB_var.flatten(), x_C2KdN_var.flatten(), x_MEMB_var.flatten()]
else:
    vars = [0.0, 0.0, x_RSOM_var.flatten(), x_C2KB_var.flatten()]

for i, x_sol in enumerate(sols):
    curve = normalization*x_sol.flatten() + (len(sols)-i)*0.2
    if (i==0):
        plt.plot(omegas, curve, color=colors[i], marker=markers[i], markevery=11+3*i, ms=10, alpha=0.5, ls ='--')
    else:
        plt.plot(omegas, curve, color=colors[i], marker=markers[i], markevery=11+3*i, ms=10)
    if (i>1): # There are no error estimates to plot for Backus-Gilbert methods
        plt.fill_between(omegas,
                         normalization*(curve - 2*np.sqrt(vars[i])),
                         normalization*(curve + 2*np.sqrt(vars[i])),
                         color=colors[i], alpha=0.5)
plt.plot(omegas, normalization*mu, color='black', ls='--', marker='x', markevery=4, label='prior')
plt.plot(omegas, normalization*x, color='black', marker='o', markevery=4, label='solution')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.xlabel(r'$\omega$ [unitless]', fontsize=16)
plt.ylim([3.5,7.0])
#plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1.0)) # outside the box
plt.savefig('imgs/gaussianInversion_allEstimates.pdf', bbox_inches='tight', format='pdf')


# Plot RSOM Diagnostics
plt.figure()
plt.plot(omegas[3:], normalization*x[3:], color='black', label='solution', marker='o', markevery=4)
for i, sol in enumerate(RSOM_proposedsolutions):
    if i == RSOM_choice:
        plt.plot(omegas, normalization*sol, label=f'BEST, $N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', color='darkorange', marker='h', markevery=4)
    else:
        plt.plot(omegas, normalization*sol, label=f'$N_c={i+Nkernels_min:d}$, MSE={RSOM_chi2[i]:.1e}', alpha=0.5)
plt.xlabel(r'$\omega$ [unitless]', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(loc='lower left', fontsize=12)
plt.title('RSOM solutions across number of kernels', fontsize=16)
plt.savefig('imgs/gaussianInversion_dictionarylearning_sols.pdf', bbox_inches='tight', format='pdf')

# Plot Backus-Gilbert Diagnostics
plt.figure()
plt.title('Backus-Gilbert Smearing Functions',  fontsize=16)
step=len(qlist_BGM)//4
dw = omegas[1] - omegas[0]
for index, q, c in zip( range(0,Nomega,step), qlist_BGM[::step], ['r','b','g','c'] ):
    plt.plot( omegas, (q.T@A)[0], c=c, marker='x', markevery=33, label=r'$\omega$=%.2f'%omegas[index])
plt.xlabel(r'$\omega$', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(fontsize=14)
plt.savefig('imgs/gaussianInversion_BG_smearingFxns.pdf', bbox_inches='tight', format='pdf')

plt.figure()
plt.title(r'Backus-Gilbert Objective Function',  fontsize=16)
lams = 10**np.linspace(np.log10(1e-3), np.log10(.999), 100)
for index, objfxn in enumerate(obj_list_BGM):
    if(index % step==0):
        plt.plot(lams, objfxn, label=r'$\omega$=%.2f'%omegas[index])
        plt.scatter(lams[np.argmax(objfxn)], np.max(objfxn))
plt.xlabel(r'regularization param $\lambda$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(fontsize=14)
plt.savefig('imgs/gaussianInversion_BG_objFxns.pdf', bbox_inches='tight', format='pdf')

# Plot Gaussian Backus-Gilbert Diagnostics
plt.figure()
plt.title(r'Gaussian Backus-Gilbert Smearing Function $\sigma=0.20$',  fontsize=16)
step=len(glist_GBGM)//5
dw = omegas[1] - omegas[0]
for smear, g, c in zip(smearing_list_GBGM[::step], glist_GBGM[::step], ['r','b','g','c'] ):
    print(dw*np.sum(smear), dw*np.sum( (g.T@A)[0] ))
    plt.plot( omegas, smear, c=c, ls='--', markevery=55, label=r'target Gaussian $\omega$=%.2f'%omegas[index])
    plt.plot( omegas, (g.T@A)[0], c=c, markevery=33, label=r'$\omega$=%.2f'%omegas[index])
plt.xlabel(r'$\omega$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(fontsize=14)
plt.savefig('imgs/gaussianInversion_GBG_smearingFxns.pdf', bbox_inches='tight', format='pdf')

plt.figure()
plt.title('Gaussian Backus-Gilbert Objective Function',  fontsize=16)
for index, objfxn in enumerate(obj_list_GBGM):
    if(index % step==0):
        plt.plot(lams, objfxn, label=r'$\omega$=%.2f'%omegas[index])
        plt.scatter(lams[np.argmax(objfxn)], np.max(objfxn))
plt.xlabel(r'regularization param $\lambda$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(fontsize=14)
plt.savefig('imgs/gaussianInversion_GBG_objFxns.pdf', bbox_inches='tight', format='pdf')


# Plot Entropy Diagnostics
# 1) proposed solutions
plt.figure()
plt.plot(x, color='black', alpha=0.7,  marker='o', markevery=17, label='solution')
for index, x in enumerate(MEMB_proposedsolutions):
    if( MEMB_acceptance_arr[index] ):
        plt.plot(x, alpha=0.7, c='cyan', label='%.2e'%MEMB_P[index,0])
    else:
        plt.plot(x, alpha=0.1, c='black', label='%.2e'%MEMB_P[index,0])
#plt.plot(x_MEMB, marker='s', markevery=29, ms=10, label='MEM Bryan estimate')
plt.ylim([0.005,0.015])
plt.legend(fontsize=7)
plt.savefig('imgs/gaussianInversion_MEMB_proposedSols.pdf', bbox_inches='tight', format='pdf')

if (dN == True):
    plt.figure()
    plt.plot(x, color='black', alpha=0.7,  marker='o', markevery=17, label='solution')
    for index, x in enumerate(MEMdN_proposedsolutions):
        if( MEMdN_acceptance_arr[index] ):
            plt.plot(x, alpha=0.7, c='cyan', label='%.2e'%MEMdN_P[index,0])
        else:
            plt.plot(x, alpha=0.1, c='black', label='%.2e'%MEMdN_P[index,0])
    #plt.plot(x_MEMB, marker='s', markevery=29, ms=10, label='MEM Bryan estimate')
    plt.ylim([0.005,0.015])
    plt.legend(fontsize=7)
    plt.savefig('imgs/gaussianInversion_MEMdN_proposedSols.pdf', bbox_inches='tight', format='pdf')

"""
# 2) Posterior plots
plt.figure()
for P, label, color, marker in zip([MEMB_P, MEMdN_P],
                           [r'Bryan $P(\alpha| b, \mu )$', r'dual N. $P(\alpha| b, \mu )$'],
                           ['blue', 'red'],
                           ['s', 'o']):
    prob = np.exp(np.sum(P[:,1:], axis=1) ) #/P_MEMdN[start:,0]
    plt.plot(P[:,0], prob / np.sum(prob), label=label, color=color, marker=marker, ms=5)

for P, acceptance_arr, color in zip([MEMB_P, MEMdN_P],
                                    [MEMB_acceptance_arr, MEMdN_acceptance_arr],
                                    ['blue', 'red']):
    start = 0; end=-1;
    for k in range(1,len(acceptance_arr)):
        if(acceptance_arr[k] and not acceptance_arr[k-1]):
            start=k
        if(acceptance_arr[k-1] and not acceptance_arr[k]):
            end=k
    print(start, end)
    plt.axvline(P[start,0], color=color,  ls='--')
    plt.axvline(P[end,0], color=color,  ls='--')
    plt.axvspan(P[start,0], P[end,0], color=color, alpha=.15)

chi2_max = np.amax(C2KB_chi2[:,1])
chi2_min = np.amin(C2KB_chi2[:,1])
tmp = (C2KB_chi2[:,1] - chi2_min)/(chi2_max - chi2_min)
plt.plot(C2KB_chi2[:,0], tmp, label=r'normalized $\chi^2(\alpha)$', c='g')

plt.axvline(best_alphas[0], color='g',  ls='--')
plt.axvline(best_alphas[-1], color='g',  ls='--')
plt.axvspan(best_alphas[0], best_alphas[-1], color='green', alpha=0.15)

plt.legend(fontsize=14)
plt.title('Comparison of entropic regularization weight techniques', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\alpha$', fontsize=16)
plt.xscale('log')
plt.savefig('imgs/gaussianInversion_entropic_alpha_selection.pdf', bbox_inches='tight', format='pdf')
print('printed images to imgs/')
"""
