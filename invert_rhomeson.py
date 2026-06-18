import numpy as np
from numpy.linalg import norm, svd
import src.MEMdualNewton as MEMdNpy
import src.MEMBryan as MEMBpy

import src.C2KBryan as C2KBpy
import src.C2KdualNewton as C2KdNpy

import src.BackusGilbert as BGpy
import src.GaussianBackusGilbert as GBGpy
import src.RSOM_langevin as RSOMpy
import matplotlib.pyplot as plt
from matplotlib import cm
colormap = cm.viridis  # You can change to 'plasma', 'jet', etc.
pi=np.pi
import time

# ------------------------------------- #
# ---- rho-meson Spectral Function ---- #
# ------------------------------------- #
# This problem is taken from realistic rho-meson example given in
# Asakawa, M., Y. Nakahara, and T. Hatsuda.'s
# "Maximum entropy analysis of the spectral functions in lattice QCD."
# Progress in Particle and Nuclear Physics 46.2 (2001): 459-508.
Nomega = 350; Ntau = 30; start_index=8# we shortened the omega range.
GeV_to_invfm = 1./5.068
fm_to_invGeV = 5.068
dw = 0.01
omegas = dw*np.array(range(Nomega)[start_index:]) # units GeV
dtau = fm_to_invGeV*0.085 # units 0.085 fm -> 1/GeV
taus = dtau*np.array(range(Ntau))

mass_rho=0.77 # rho mass in GeV
mass_pion=0.14 # pion mass in GeV
g2_rhopionpion=5.45**2 # rho-pion-pion vertex/interaction strength (squared)
w0 = 1.3 # GeV
delta = 0.2 # GeV
alpha_s = 0.3 # strong force coupling parameter
# define rho particle decay rate
GG_rho = g2_rhopionpion / (48*pi) * mass_rho * ( 1 - 4*mass_pion**2 / omegas**2 )**(3/2) * np.heaviside(omegas - 2*mass_pion , 0.5)
GG_rho [ np.isnan(GG_rho) ] =0.0
# define lorentzian peak around the pole
F2_rho = mass_rho**2 / g2_rhopionpion
pole = F2_rho* GG_rho * mass_rho / (( omegas**2 - mass_rho**2 )**2 + GG_rho**2 * mass_rho**2 )
cut = 1/(8*pi)*(1 + alpha_s/pi) / (1 + np.exp( (w0 - omegas) / delta))
x =  2 / pi * (pole + cut) #* omegas**2
#x /= np.sum(x)
mu = 0.0257*np.ones((Nomega-start_index,1))
#mu /= np.sum(mu)


# ------------------------------- #
# ---- Create Synthetic Data ---- #
# ------------------------------- #
noise_level=0.01
noiselevel='0p01'
Nsamples=100
b_samples = np.zeros((Nsamples,Ntau))
EE, TT = np.meshgrid(omegas, taus)
A = TT*EE
for i in range(len(taus)):
    for j in range(len(omegas)):
        A[i,j]= np.exp(-A[i,j]) * omegas[j]**2

#generate data
b_clean = A @ x
delta_tau = 1e-2
if (0 == 1):
    stdev = noise_level * b_clean * (taus+delta_tau) /(taus[1]-taus[0])
    for i in range(Nsamples):
        # old way
        #data[i] = data0 * (1 + noise_level * np.reshape(taus,(-1,)) /dtau * randn(Ntau,) )
        #noisetest = b_clean * noise_level * np.reshape(taus,(-1,)) /dtau #* randn(Ntau,)
        noise = np.random.normal(np.zeros((Ntau,)), stdev)
        b_samples[i] = b_clean + noise

    b_avg = np.average(b_samples, axis=0)
    C_iid = np.diag(np.var(b_samples, axis=0, ddof=1)/Nsamples)
    #C_full = (b_samples-b_avg).T @ (b_samples-b_avg) / Nsamples
    np.savez('data/rho-meson_noise'+noiselevel+'.npz', A=A, x=x, b_avg=b_avg, Cov=C_iid, mu=mu, taus=taus, omegas=omegas) # .npy extension is added if not given

# Load data and parse arugments
npz = np.load('data/rho-meson_noise'+noiselevel+'.npz', allow_pickle=True)
x=npz['x']; mu=npz['mu']; A=npz['A']; b_avg=npz['b_avg']; Cov=npz['Cov']; taus=npz['taus']; omegas=npz['omegas'];
b_clean = A@x
b_err = np.sqrt(np.diag(Cov))
normalization = b_avg[0]

# asses the data
res = np.reshape(b_avg - b_clean, (-1,1))
b0_chi2 = (res.T@np.linalg.inv(Cov)@res)[0,0]/Ntau
print('chi-sq of true data', b0_chi2)
print('relative error in b', np.linalg.norm(b_err) / np.linalg.norm(b_avg) )
print('A condition number', np.linalg.cond(A))

# create normalized quantitites for routine
grad_scale = 1e-4/np.linalg.norm(b_err/normalization) # what counts as a small gradient? Ax-b = 1e-4 scaled by the error
print('grad_scale', grad_scale)
b_input = np.reshape( b_avg/normalization, (-1,1))
C_iid = np.diag(np.diag(Cov)/(normalization**2))
m = mu/normalization
print('x sum', np.sum(x), 'b_avg[0]', normalization, (A@x)[0] )
print('x sum / normalized', np.sum(x)/normalization, 'm sum', np.sum(m))

# ---------------------------- #
# Do Data Inversion with noise #
# ---------------------------- #
print('data inversions with noise')
# Backus-Gilbert Methods
#x_BGM, qlist_BGM, obj_list_BGM = BGpy.BGM_solve(A, b_input, C_iid, omegas )
#x_GBGM, glist_GBGM, obj_list_GBGM, smearing_list_GBGM = GBGpy.GaussianBGM_solve(A, b_input, C_iid, omegas, taus, sigma=0.15, lam_input=0.9)

# Entropic Methods
Nalpha=31
alpha_min=5e0
alpha_max=1e3
rtol=1e-8
atol=1e-8
tmp = MEMdNpy.MEMdN_solve(A, b_input, C_iid, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                          rtol=rtol, atol=atol)
x_MEMdN, x_MEMdN_var, proposed_solutions_MEMdN, P_MEMdN, acceptance_arr_MEMdN = tmp

tmp = MEMBpy.MEMBryan_solve(A, b_input, C_iid, m, alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha) #, max_iter=10)
x_MEMB, x_MEMB_var, proposed_solutions_MEMB, P_MEMB, acceptance_arr_MEMB = tmp

print('primal', P_MEMB)
print('primal', acceptance_arr_MEMB)
print('dual', P_MEMdN)
print('dual', acceptance_arr_MEMdN)

# Entropic Regularization with Chi-2-kink algorithm for alpha selection
C2K_min = 2.5
C2K_max = 3.0
NC2K = 10
alpha_min=2e-2
alpha_max=1e9
tmp = C2KBpy.C2KBryan_solve(A, b_input, C_iid, m,
                            alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                            C2K_min=C2K_min, C2K_max=C2K_max, NC2K=NC2K)
x_C2KB, x_C2KB_var, C2KB_proposedsolutions, C2KB_chi2, C2KB_best_alphas = tmp
print('shape C2KB', np.shape(x_C2KB), np.shape(x_C2KB_var))

tmp = C2KdNpy.C2KdN_solve(A, b_input, C_iid, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                          C2K_min=C2K_min, C2K_max=C2K_max, NC2K=NC2K,
                          rtol=rtol, atol=atol)

x_C2KdN, x_C2KdN_var, C2KdN_proposedsolutions, C2KdN_chi2, C2KdN_best_alphas = tmp
print('shape C2KdN', np.shape(x_C2KdN), np.shape(x_C2KdN_var))


# stochastic optimization where the coefficients are set with regularized optimization
Nkernels_min=1
Nkernels_max=4
tmp = RSOMpy.RSOM_solve(A, b_input, C_iid, mu, omegas, taus,
                        step_size=1e-2, beta_min=1e2, beta_max=1e5, Nbeta=200,
                        Npt=5000, Nmh=10, burn_in=300, Nsamples=20000, # MH solution collection
                        genetic_flag=1, p_reset=0.001, # whether to use genetic reset
                        Nkernels_min=Nkernels_min, Nkernels_max=Nkernels_max, Nkernels_step=1,
                        grad_scale=grad_scale) #1e-1)

x_RSOM, x_RSOM_var, RSOM_proposedsolutions, RSOM_chi2, RSOM_choice = tmp

#                        step_size=5e-4, beta_min=1e2, beta_max=1e5, Nbeta=200,
#                        Npt=5000, Nmh=10, burn_in=1, Nsamples=100, # MH solution collection
#                        genetic_flag=1, p_reset=0.001, # whether to use genetic reset


# ------------------------------- #
# Do Data Inversion without noise #
# ------------------------------- #
#'nl' = noiseless
print('data inversions without noise')
tmp = MEMBpy.MEMBryan_solve(A, b_clean/(A @ x)[0], C_iid, m,
                            alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                            max_iter=10)
x_MEMB_nl, x0_MEMB_var_nl, proposed_solutions_MEMB_nl, P_MEMB_nl, acceptance_arr_MEMB_nl = tmp

tmp = MEMdNpy.MEMdN_solve(A, b_clean/(A @ x)[0], C_iid, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                          rtol=rtol, atol=atol)
x_MEMdN_nl, x0_MEMdN_var_nl, proposed_solutions_MEMdN_nl, P_MEMdN_nl, acceptance_arr_MEMdN_nl = tmp

# ------------------------------ #
# PLOT PRIMAL ALPHA SOLUTIONS    #
# also construct error values    #
# when looping through solutions #
# ------------------------------ #
deltab_norm = norm( (b_avg-b_clean.flatten())/normalization )
print('delta b norm', deltab_norm)
U, S, Vh = svd(A)
A_norm = S[0]
print('A norm', A_norm)
U, S, Vh = svd(C_iid)
C_iid_norm = S[0]
print('C norm', C_iid_norm)

xmax=2.5
plt.figure()
plt.title("Accepted solutions from Bryan's alg", fontsize=16)
Bryan_deltax_norm_list = [];
BdN1_deltax_norm_list = [];
BdN2_deltax_norm_list = [];
BdN3_deltax_norm_list = [];
BdN4_deltax_norm_list = [];
Bryan_RHS_norm_list = [];
j=0; # index for color map
n_curves = int(np.sum(acceptance_arr_MEMB))
colors = colormap(np.linspace(0, 1, n_curves))
for i in range(len(acceptance_arr_MEMB)):
    BdN1_deltax_norm_list.append( norm( proposed_solutions_MEMB[i] - proposed_solutions_MEMdN[i]) )
    BdN2_deltax_norm_list.append( norm( proposed_solutions_MEMB_nl[i] - proposed_solutions_MEMdN_nl[i]) )
    BdN3_deltax_norm_list.append( norm( proposed_solutions_MEMB[i] - proposed_solutions_MEMdN_nl[i]) )
    BdN4_deltax_norm_list.append( norm( proposed_solutions_MEMB_nl[i] - proposed_solutions_MEMdN[i]) )

    Bryan_deltax_norm_list.append( norm( proposed_solutions_MEMB[i] - proposed_solutions_MEMB_nl[i]) )
    Bryan_RHS_norm_list.append(  deltab_norm*A_norm/C_iid_norm/P_MEMB[i,0] )
    if (acceptance_arr_MEMB[i]):
        if ( j ==0 or j == n_curves-1):
            plt.plot(omegas, normalization*proposed_solutions_MEMB[i], label=r'$\alpha=%.2e$'%P_MEMB[i,0], color=colors[j] )
        else:
            plt.plot(omegas, normalization*proposed_solutions_MEMB[i], color=colors[j] )
        j+=1
plt.text(xmax/20, 0.054, r'$\sigma=$'+str(noise_level), fontsize=16)
plt.legend(loc='upper right', fontsize=14) #bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xlabel(r'$\omega$ (GeV)', fontsize=16)
plt.xlim([0,xmax])
plt.ylim([0,0.06])
plt.tick_params(axis='both', labelsize=16)
print('imgs/rho-meson_proposals-primal_noise'+noiselevel+'_priorflat.pdf')
plt.savefig('imgs/rho-meson_proposals-primal_noise'+noiselevel+'_priorflat.pdf', bbox_inches='tight', format='pdf')

# ------------------------------ #
# PLOT DUAL ALPHA SOLUTIONS      #
# also construct error values    #
# when looping through solutions #
# ------------------------------ #
plt.figure()
plt.title('Accepted solutions from dual Newton alg.', fontsize=16)
dNewton_deltax_norm_list = [];
dNewton_RHS_norm_list = [];
j=0; # index for color map
n_curves = int(np.sum(acceptance_arr_MEMdN))
colors = colormap(np.linspace(0, 1, n_curves))
for i in range(len(acceptance_arr_MEMdN)):
    dNewton_deltax_norm_list.append(norm( proposed_solutions_MEMdN[i] - proposed_solutions_MEMdN_nl[i]))
    dNewton_RHS_norm_list.append(deltab_norm*A_norm/C_iid_norm/P_MEMdN[i,0])
    if (acceptance_arr_MEMdN[i]):
        print(j, n_curves-1)
        if ( j ==0 or j == n_curves-1):
            plt.plot(omegas, normalization*proposed_solutions_MEMdN[i], label=r'$\alpha=%.2e$'%P_MEMdN[i,0], color=colors[j] )
        else:
            plt.plot(omegas, normalization*proposed_solutions_MEMdN[i], color=colors[j] )
        j+=1
plt.text(xmax/20, 0.054, r'$\sigma=$'+str(noise_level), fontsize=16)
plt.legend(loc='upper right', fontsize=14) #bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xlabel(r'$\omega$ (GeV)', fontsize=16)
plt.xlim([0,xmax])
plt.ylim([0,0.06])
plt.tick_params(axis='both', labelsize=16)
print('imgs/rho-meson_proposals-dual_noise'+noiselevel+'_priorflat.pdf')
plt.savefig('imgs/rho-meson_proposals-dual_noise'+noiselevel+'_priorflat.pdf', bbox_inches='tight', format='pdf')

val = normalization
plt.figure()
plt.plot(omegas, x, label='solution', c='black', marker='o', markevery=35)
plt.plot(omegas, mu, label='prior',  c='black', marker='x', markevery=17, alpha=0.75, ls='--')
#plt.plot(omegas, val*x_BGM, label='BGM' )
#plt.plot(omegas, val*x_GBGM, label='Gaussian BGM', color='g')
plt.plot(omegas, val*x_RSOM, label='RSOM', c='darkorange', marker='h', markevery=67 )
#plt.plot(omegas, val*RSOM_proposedsolutions.T, label='RSOM', c='black', alpha=0.5 )
plt.plot(omegas, val*(x_RSOM+2*np.sqrt(x_RSOM_var)), c='darkorange', ls='--', alpha=0.5, label='error')
plt.plot(omegas, val*(x_RSOM-2*np.sqrt(x_RSOM_var)), c='darkorange', ls='--', alpha=0.5)
plt.plot(omegas, val*x_C2KB, label=r'$\chi^2$k-B', c='blue', marker='s', markevery=55, ms=5)
plt.plot(omegas, val*(x_C2KB+2*np.sqrt(x_C2KB_var)), c='blue', ls='--', alpha=0.5, label='error')
plt.plot(omegas, val*(x_C2KB-2*np.sqrt(x_C2KB_var)), c='blue', ls='--', alpha=0.5)
plt.plot(omegas, val*x_C2KdN, label=r'$\chi^2$k-dN', c='red', marker='^', markevery=42, ms=5)
plt.plot(omegas, val*(x_C2KdN+2*np.sqrt(x_C2KdN_var)), c='red', ls='--', alpha=0.5, label='error')
plt.plot(omegas, val*(x_C2KdN-2*np.sqrt(x_C2KdN_var)), c='red', ls='--', alpha=0.5)
plt.xlabel(r'$\omega$ (GeV)', fontsize=16)
plt.xlim([omegas[2],xmax])
plt.ylim([0,0.15])
plt.legend(fontsize=15, loc='upper right')
plt.tick_params(axis='both', labelsize=16)
plt.title('Inverse problem solutions', fontsize=16)
print('imgs/rho-meson_reconstructions_noise'+noiselevel+'_prior-flat.pdf')
plt.savefig('imgs/rho-meson_reconstructions_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# Plot data
plt.figure()
plt.title('Transformed solutions compared to data', fontsize=16)
plt.errorbar(taus, b_avg, b_err, color='black', label='avg +/- err')
plt.plot(taus, A@x, color='black', label='$b^0$'+ r', $\tilde{\chi}^2 = $'+f'{b0_chi2:.3f}', alpha=0.5)
sols = [x_RSOM, x_C2KB, x_C2KdN]
colors=['darkorange', 'blue', 'red']
markers=['h', 's', '^']
labels=['RSOM', r'$\chi^2$k-B', r'$\chi^2$k-dN']
for i, x_sol in enumerate(sols):
    x_sol  = np.reshape(x_sol, (-1, 1))
    res = A@x_sol - b_input
    chi2 = (res.T@np.diag(1./np.diag(C_iid))@res/Ntau)[0,0]
    plt.plot(taus, val*A@x_sol, color=colors[i], marker=markers[i], markevery=3+i, ms=10, label=labels[i] + r', $\tilde{\chi}^2 = $'+f'{chi2:.2f}')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\tau$ (1/GeV)', fontsize=16)
plt.yscale('log')
plt.legend(fontsize=15)
plt.savefig('imgs/rho-meson_syntheticdata_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# Plot RSOM Diagnostics
plt.figure()
plt.plot(omegas[3:], x[3:], color='black', label='solution', marker='.', markevery=4)
for i, sol in enumerate(RSOM_proposedsolutions):
    print('plotting Nkernels', i)
    if i == RSOM_choice:
        print(i, 'is the best')
        plt.plot(omegas, val*x_RSOM, label=f'RSOM, $N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', color='darkorange', marker='h', markevery=4)
    else:
        print(i, 'is not the best')
        plt.plot(omegas, val*sol, label=f'$N_c={i+Nkernels_min:d}$, ' + r'$\tilde{\chi}^2$'+f'={RSOM_chi2[i]:.1e}', alpha=0.5)
plt.xlabel(r'$\omega$ (GeV)', fontsize=14)
plt.legend(loc='upper right', fontsize=15)
plt.tick_params(axis='both', labelsize=16)
#plt.tick_params(axis='both', which='major', labelsize=10)
#plt.tick_params(axis='both', which='minor', labelsize=8)
plt.ylim([0, 0.15])
plt.title('RSOM solutions across number of kernels', fontsize=16)
print('saved figure: imgs/rho-meson_dictionarylearning_sols_noise'+noiselevel+'.pdf')
plt.savefig('imgs/rho-meson_dictionarylearning_sols_noise'+noiselevel+'.pdf', bbox_inches='tight', format='pdf')

# ----------------------------- #
# PLOT PERTURBATIVE ERROR BOUND #
# ----------------------------- #
plt.figure()
#plt.plot(P_MEMdN[:,0], 3e-6*np.array(dNewton_RHS_norm_list),  c='black', ls=':', ms=5, label=r"$3 \times 10^{-6} (1/\alpha) || C^{-1}||$ $||A||$ $||b - b'||$")
plt.plot(P_MEMdN[:,0], dNewton_RHS_norm_list, c='black', label=r"$|| C^{-1}|| \, ||A|| \, ||b - b'|| \, \alpha^{-1}$")
plt.plot(P_MEMdN[:,0], 1e-4*P_MEMdN[:,0]**(-.5),  c='black', ls='--', ms=5,  label=r'$10^{-4} \, \alpha^{-1/2}$')
plt.plot(P_MEMdN[:,0], Bryan_deltax_norm_list,  c='blue', marker='s', markevery=2, ms=8, label="Bryan ||x - x'||")
plt.plot(P_MEMdN[:,0], dNewton_deltax_norm_list, c='red', marker='^', ms=7, label="dual Newton ||x - x'||")
plt.plot(P_MEMdN[:,0], BdN1_deltax_norm_list,  c='purple', ls='-.', label=r"$||x_{Bryan} - x_{dual\,  N.}||$", alpha=0.5)
plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$\alpha$', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.savefig('imgs/rho-meson_dataperturbationerrorbounds_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# --------------------------- #
# PLOT POSTERIOR DISTRIBUTION #
# --------------------------- #
plt.figure()
for P, label, color, marker in zip([P_MEMB,P_MEMdN],
                           [r'Bryan $P(\alpha| b, \mu )$', r'dual Newton $P(\alpha | b, \mu )$'],
                           ['blue', 'red'],
                           ['s','^']):
    prob = np.exp(np.sum(P[:,1:], axis=1) ) #/P_MEMdN[start:,0]
    plt.plot(P[:,0], prob / np.sum(prob), label=label, color=color, marker=marker, ms=5)

for P, acceptance_arr, color in zip([P_MEMB, P_MEMdN],[acceptance_arr_MEMB,acceptance_arr_MEMdN],['blue', 'red']):
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

plt.legend(fontsize=14)
plt.title('Solution weight via Bayesian posterier', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
#plt.ylim([0,0.16])
plt.xlabel(r'$\alpha$', fontsize=16)
plt.xscale('log')
print('imgs/rho-meson_posteriorprob_noise'+noiselevel+'_prior-flat.pdf')
plt.savefig('imgs/rho-meson_posteriorprob_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')


