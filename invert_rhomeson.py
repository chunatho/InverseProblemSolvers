import numpy as np
from numpy.linalg import norm, svd
import src.MEMdualNewton as MEMdNpy
import src.MEMBryan as MEMBpy
import src.BackusGilbert as BGpy
import src.GaussianBackusGilbert as GBGpy
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
Nomega = 600; Ntau = 30;
GeV_to_fminv = 1./5.068
fm_to_GeVinv = 5.068
print( (4 - 12./Nomega)/ (Nomega - 3) )
omegas = np.linspace(12./Nomega, 6, Nomega-3)  # units GeV
dw = omegas[1] - omegas[0] # units GeV
dtau = fm_to_GeVinv*0.085 # units 0.085 fm -> 1/GeV
taus = dtau*np.array(range(Ntau))
print('dw',dw, 'dtau', dtau)

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
mu = 0.0257*np.ones((Nomega-3,1))
#mu /= np.sum(mu)


# ------------------------------- #
# ---- Create Synthetic Data ---- #
# ------------------------------- #
noise_level=0.001
noiselevel='0p001'
Nsamples=1000
b_samples = np.zeros((Nsamples,Ntau))
EE, TT = np.meshgrid(omegas, taus)
A = TT*EE
for i in range(len(taus)):
    for j in range(len(omegas)):
        A[i,j]= np.exp(-A[i,j]) * omegas[j]**2

#generate data
b_clean = A @ x
if (1 == 0):
    stdev = np.sqrt( noise_level * b_clean * (taus+1e-6) /(taus[1]-taus[0]) )
    for i in range(Nsamples):
        # old way
        #data[i] = data0 * (1 + noise_level * np.reshape(taus,(-1,)) /dtau * randn(Ntau,) )
        #noisetest = b_clean * noise_level * np.reshape(taus,(-1,)) /dtau #* randn(Ntau,)
        noise = np.random.normal(np.zeros((Ntau,)), stdev)
        b_samples[i] = b_clean + noise

    b_avg = np.average(b_samples, axis=0)
    b_std = np.std(b_samples, axis=0)
    print('stdev', b_std)
    np.savez('data/rho-meson_noise'+noiselevel+'.npz', A=A, x=x, b_avg=b_avg, b_std=b_std, mu=mu, taus=taus, omegas=omegas) # .npy extension is added if not given

npz = np.load('data/rho-meson_noise'+noiselevel+'.npz', allow_pickle=True)
x=npz['x']; mu=npz['mu']; A=npz['A']; b_avg=npz['b_avg']; b_std=npz['b_std']; taus=npz['taus']; omegas=npz['omegas'];
print('x sum',np.sum(x), 'b_avg[0]', b_avg[0], (A@x)[0], 'expected normalization', np.sum(x) / b_avg[0] )
#C = (b_samples-b_avg).T @ (b_samples-b_avg) / Nsamples

b=np.reshape( b_avg/b_avg[0], (-1,1))
C = np.diag( (b_std/b_avg[0])**2 )
m = mu/b_avg[0]
#print('sum x', np.sum(x)/b_avg[0])
#print('sum m', np.sum(m))

# ---------------------------- #
# Do Data Inversion with noise #
# ---------------------------- #
# Entropic Methods
print('data inversions with noise')
tmp = MEMdNpy.MEMdN_solve(A, b, C, m,
                          alpha_min=1e1, alpha_max=1e6, Nalpha=31,
                          rtol=1e-6, atol =1e-6, xrtol=1e-6, xatol=1e-6)
x_dMEM, x_dMEM_var, proposed_solutions_dMEM, P_dMEM, acceptance_arr_dMEM = tmp

tmp = MEMBpy.MEMBryan_solve(A, b, C, m, alpha_min=1e1, alpha_max=1e6, Nalpha=31, max_iter=10)
x_pMEM, x_pMEM_var, proposed_solutions_pMEM, P_pMEM, acceptance_arr_pMEM = tmp

print('primal', P_pMEM)
print('primal', acceptance_arr_pMEM)
print('dual', P_dMEM)
print('dual', acceptance_arr_dMEM)

# Backus-Gilbert Methods
#x_BG, qlist_BG, obj_list_BG = BGpy.BG_solve(A, b, C, omegas ) # TO DO: Implement Smoothened BGM to set lam parameter
#x_GBG, glist_GBG, obj_list_GBG, smearing_list_GBG = GBGpy.GaussianBG_solve(A, b, C, omegas, taus, sigma=1)

# ------------------------------- #
# Do Data Inversion without noise #
# ------------------------------- #
#'nl' = noiseless
print('data inversions without noise')
tmp = MEMBpy.MEMBryan_solve(A, b_clean/(A @ x)[0], C, m, alpha_min=1e1, alpha_max=1e6, Nalpha=31, max_iter=10)
x_pMEM_nl, x0_pMEM_var_nl, proposed_solutions_pMEM_nl, P_pMEM_nl, acceptance_arr_pMEM_nl = tmp

tmp = MEMdNpy.MEMdN_solve(A, b_clean/(A @ x)[0], C, m,
                          alpha_min=1e1, alpha_max=1e6, Nalpha=31,
                          rtol=1e-6, atol =1e-6, xrtol=1e-6, xatol=1e-6)
x_dMEM_nl, x0_dMEM_var_nl, proposed_solutions_dMEM_nl, P_dMEM_nl, acceptance_arr_dMEM_nl = tmp

# ------------------------------ #
# PLOT PRIMAL ALPHA SOLUTIONS    #
# also construct error values    #
# when looping through solutions #
# ------------------------------ #
deltab_norm = norm( (b_avg-b_clean)/b_avg[0] )
print('delta b norm', deltab_norm)
U, S, Vh = svd(A)
A_norm = S[0]
print('A norm', A_norm)
U, S, Vh = svd(C)
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
n_curves = int(np.sum(acceptance_arr_pMEM))
colors = colormap(np.linspace(0, 1, n_curves))
for i in range(len(acceptance_arr_pMEM)):
    BdN1_deltax_norm_list.append( norm( proposed_solutions_pMEM[i] - proposed_solutions_dMEM[i]) )
    BdN2_deltax_norm_list.append( norm( proposed_solutions_pMEM_nl[i] - proposed_solutions_dMEM_nl[i]) )
    BdN3_deltax_norm_list.append( norm( proposed_solutions_pMEM[i] - proposed_solutions_dMEM_nl[i]) )
    BdN4_deltax_norm_list.append( norm( proposed_solutions_pMEM_nl[i] - proposed_solutions_dMEM[i]) )

    Bryan_deltax_norm_list.append( norm( proposed_solutions_pMEM[i] - proposed_solutions_pMEM_nl[i]) )
    Bryan_RHS_norm_list.append(  deltab_norm*A_norm/C_iid_norm/P_pMEM[i,0] )
    if (acceptance_arr_pMEM[i]):
        if ( j ==0 or j == n_curves-1):
            plt.plot(omegas, b_avg[0]*proposed_solutions_pMEM[i], label=r'$\alpha=%.2e$'%P_pMEM[i,0], color=colors[j] )
        else:
            plt.plot(omegas, b_avg[0]*proposed_solutions_pMEM[i], color=colors[j] )
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
n_curves = int(np.sum(acceptance_arr_dMEM))
colors = colormap(np.linspace(0, 1, n_curves))
for i in range(len(acceptance_arr_dMEM)):
    dNewton_deltax_norm_list.append(norm( proposed_solutions_dMEM[i] - proposed_solutions_dMEM_nl[i]))
    dNewton_RHS_norm_list.append(deltab_norm*A_norm/C_iid_norm/P_dMEM[i,0])
    if (acceptance_arr_dMEM[i]):
        print(j, n_curves-1)
        if ( j ==0 or j == n_curves-1):
            plt.plot(omegas, b_avg[0]*proposed_solutions_dMEM[i], label=r'$\alpha=%.2e$'%P_dMEM[i,0], color=colors[j] )
        else:
            plt.plot(omegas, b_avg[0]*proposed_solutions_dMEM[i], color=colors[j] )
        j+=1
plt.text(xmax/20, 0.054, r'$\sigma=$'+str(noise_level), fontsize=16)
plt.legend(loc='upper right', fontsize=14) #bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xlabel(r'$\omega$ (GeV)', fontsize=16)
plt.xlim([0,xmax])
plt.ylim([0,0.06])
plt.tick_params(axis='both', labelsize=16)
print('imgs/rho-meson_proposals-dual_noise'+noiselevel+'_priorflat.pdf')
plt.savefig('imgs/rho-meson_proposals-dual_noise'+noiselevel+'_priorflat.pdf', bbox_inches='tight', format='pdf')

val = b_avg[0]
plt.figure()
plt.plot(omegas, x, label='exact', c='black', marker='o', markevery=35)
plt.plot(omegas, mu, label='prior',  c='black', marker='x', markevery=17, alpha=.5)
#plt.plot(omegas, x_BGM, label='BGM' )
#plt.plot(omegas, x_sBGM, label='sBGM' )
plt.plot(omegas, val*x_pMEM, label='Bryan', c='blue', marker='s', markevery=55, ms=5)
plt.plot(omegas, val*(x_pMEM+2*np.sqrt(x_pMEM_var)), c='blue', ls='--', label=r'error', alpha=.5)
plt.plot(omegas, val*(x_pMEM-2*np.sqrt(x_pMEM_var)), c='blue', ls='--', alpha=.5)
plt.plot(omegas, val*x_dMEM, label='dual Newton', c='red', marker='^', markevery=42, ms=5)
plt.plot(omegas, val*(x_dMEM+2*np.sqrt(x_dMEM_var)), c='red', ls='--', label=r'error', alpha=.5)
plt.plot(omegas, val*(x_dMEM-2*np.sqrt(x_dMEM_var)), c='red', ls='--', alpha=.5)
plt.xlabel(r'$\omega$ (GeV)', fontsize=16)
plt.xlim([0,xmax])
plt.ylim([0,0.15])
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)
print('imgs/rho-meson_reconstructions_noise'+noiselevel+'_prior-flat.pdf')
plt.savefig('imgs/rho-meson_reconstructions_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# ----------------------------- #
# PLOT PERTURBATIVE ERROR BOUND #
# ----------------------------- #
plt.figure()
#plt.plot(P_dMEM[:,0], 3e-6*np.array(dNewton_RHS_norm_list),  c='black', ls=':', ms=5, label=r"$3 \times 10^{-6} (1/\alpha) || C^{-1}||$ $||A||$ $||b - b'||$")
plt.plot(P_dMEM[:,0], dNewton_RHS_norm_list, c='black', label=r"$|| C^{-1}|| \, ||A|| \, ||b - b'|| \, \alpha^{-1}$")
plt.plot(P_dMEM[:,0], 1e-4*P_dMEM[:,0]**(-.5),  c='black', ls='--', ms=5,  label=r'$10^{-4} \, \alpha^{-1/2}$')
plt.plot(P_dMEM[:,0], Bryan_deltax_norm_list,  c='blue', marker='s', markevery=2, ms=8, label="Bryan ||x - x'||")
plt.plot(P_dMEM[:,0], dNewton_deltax_norm_list, c='red', marker='^', ms=7, label="dual Newton ||x - x'||")
plt.plot(P_dMEM[:,0], BdN1_deltax_norm_list,  c='purple', ls='-.', label=r"$||x_{Bryan} - x_{dual\,  N.}||$", alpha=.5)
plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$\alpha$', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.savefig('imgs/rho-meson_dataperturbationerrorbounds_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# --------------------------- #
# PLOT POSTERIOR DISTRIBUTION #
# --------------------------- #
plt.figure()
for P, label, color, marker in zip([P_pMEM,P_dMEM],
                           [r'Bryan $P(\alpha| b, \mu )$', r'dual Newton $P(\alpha | b, \mu )$'],
                           ['blue', 'red'],
                           ['s','^']):
    prob = np.exp(np.sum(P[:,1:], axis=1) ) #/P_dMEM[start:,0]
    plt.plot(P[:,0], prob / np.sum(prob), label=label, color=color, marker=marker, ms=5)

for P, acceptance_arr, color in zip([P_pMEM, P_dMEM],[acceptance_arr_pMEM,acceptance_arr_dMEM],['blue', 'red']):
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


# spectrum
#plt.plot(omegas, x )
#plt.plot(omegas, mu )
#plt.tick_params(axis='both', labelsize=16)
#plt.xlabel(r'$\omega$ (GeV)', fontsize=24)
#plt.title(r'$\rho$ ($\omega$)', fontsize=24)
#plt.savefig('LQCD-spectralfunction.pdf', bbox_inches='tight', format='pdf')

# kernel
#fig, ax = plt.subplots()
#omega, tau = np.meshgrid( omegas, taus )
#print('shape of tau', np.shape(tau), taus[-1])
#mesh = ax.pcolormesh(omega, tau, np.log(A) )
#ax.invert_yaxis()
#ax.set_title(r'log[ exp($-\tau \omega$) ]', fontsize=20)
#plt.xlabel(r'$\omega$ (GeV)', fontsize=24)
#plt.ylabel(r'$\tau (1/GeV)$', fontsize=24)
#ax.tick_params(axis='both', labelsize=16)
#plt.savefig('LQCD-Kernel.pdf', bbox_inches='tight', format='pdf')

# synthetic data
#plt.figure()
#plt.plot(taus, np.log(b_samples).T, alpha=.2)
#plt.errorbar(taus, b_avg, b_std, color='black')
#plt.xlabel(r'$\tau$ (1/GeV)', fontsize=16)
#plt.savefig('LQCD-syntheticdata.pdf')
