import numpy as np
import src.MEMBryan as MEMBpy
import src.MEMdualNewton as MEMdNpy
import matplotlib.pyplot as plt

from numpy.linalg import norm, svd
import src.BackusGilbert as BGpy
import src.GaussianBackusGilbert as GBGpy
from matplotlib import cm
colormap = cm.viridis  # You can change to 'plasma', 'jet', etc.
pi=np.pi
import time

# ------------------------- #
# ---- Double Gaussian ---- #
# ------------------------- #
# This spectrum is text case 1 from Goulko, Olga, et al.
# "Numerical analytic continuation: Answers to well-posed questions."
# Physical Review B 95.1 (2017): 014102.
Nomega = 100;
omegas = np.linspace(3.5/Nomega, 4.5, Nomega-1, endpoint=True)  # units GeV
dw = omegas[1] - omegas[0] # units GeV
Ntau = 50;
taus = np.linspace(0, 5, Ntau);  # units GeV
dtau = taus[1] - taus[0];
print(Nomega, Ntau)
print('dw', dw, 'dtau', dtau)

c1=0.62; sigma1=0.12; z1=0.74
c2=0.41; sigma2=0.064; z2=2.93
x = c1/sigma1*np.exp(-(omegas-z1)**2/2/sigma1**2 )
x += c2/sigma2*np.exp(-(omegas-z2)**2/2/sigma2**2 )
print('x', x)
mu = (np.sum(x)/len(x))*np.ones_like(x)

# ------------------------------- #
# ---- Create Synthetic Data ---- #
# ------------------------------- #
noise_level=0.001
noiselevel='0p001'
if (1 == 1): #generate data
    Nsamples=1000
    b_samples = np.zeros((Nsamples,Ntau))
    EE, TT = np.meshgrid(omegas, taus)
    A = TT*EE
    for i in range(len(taus)):
        for j in range(len(omegas)):
            A[i,j]= np.exp(-A[i,j])

    b_clean = A @ x
    stdev = noise_level * b_clean
    # redistribute the sigma vector, but keep the vector's magnitude the same.
    #weights = np.geomspace(0.5, 2., Ntau, endpoint=True)
    #weights /= np.sum(weights)
    #original_vector = noise_level*b_clean
    #weighted_vector = weights*original_vector
    #stdev = weighted_vector*np.linalg.norm(original_vector)/np.linalg.norm(weighted_vector)
    #print('unity check', np.linalg.norm(original_vector), np.linalg.norm(weighted_vector))
    #print('norm test', np.linalg.norm(stdev), np.linalg.norm(original_vector))
    for i in range(Nsamples):
        noise = np.random.normal(np.zeros((Ntau,)), stdev)
        b_samples[i] = b_clean + noise

    b_avg = np.average(b_samples, axis=0)
    b_std = np.std(b_samples, axis=0)
    print('stdev', b_std)
    np.savez('data/double-gaussian_noise'+noiselevel+'.npz', A=A, x=x, b_avg=b_avg, b_std=b_std, mu=mu, taus=taus, omegas=omegas) # .npy extension is added if not given
    #C = (b_samples-b_avg).T @ (b_samples-b_avg) / Nsamples

print('loading data saved to: double-gaussian_noise'+noiselevel+'.npz')
npz = np.load('data/double-gaussian_noise'+noiselevel+'.npz', allow_pickle=True)
x=npz['x']; mu=npz['mu']; A=npz['A']; b_avg=npz['b_avg']; b_std=npz['b_std']; taus=npz['taus']; omegas=npz['omegas'];
b_clean = A @ x
Ntau=len(taus); Nomega=len(omegas)
print('x sum',np.sum(x), 'b_avg[0]', b_avg[0], (A@x)[0], 'expected normalization', np.sum(x) / b_avg[0] )

b=np.reshape( b_avg/b_avg[0], (-1,1))
C = np.diag( (b_std/b_avg[0])**2 )
m = mu/b_avg[0]
#print('sum x', np.sum(x)/b_avg[0])
#print('sum m', np.sum(m))

# ---------------------------- #
# Do Data Inversion with noise #
# ---------------------------- #
alpha_min=6e-1
alpha_max=1.2e1
# Entropic Methods
print('data inversions with noise')
tmp = MEMdNpy.MEMdN_solve(A, b, C, m,
                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=31,
                          rtol=1e-1, atol =1e-1, xrtol=1e-1, xatol=1e-1)
x_dMEM, x_dMEM_var, proposed_solutions_dMEM, P_dMEM, acceptance_arr_dMEM = tmp
print('dual', P_dMEM)
print('dual', acceptance_arr_dMEM)

tmp = MEMBpy.MEMBryan_solve(A, b, C, m, alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=31)
x_pMEM, x_pMEM_var, proposed_solutions_pMEM, P_pMEM, acceptance_arr_pMEM = tmp
print('primal', P_pMEM)
print('primal', acceptance_arr_pMEM)
print('primal', x_pMEM)

# Backus-Gilbert Methods
#x_BG, qlist_BG, obj_list_BG = BGpy.BG_solve(A, b, C, omegas ) # TO DO: Implement Smoothened BGM to set lam parameter
#x_GBG, glist_GBG, obj_list_GBG, smearing_list_GBG = GBGpy.GaussianBG_solve(A, b, C, omegas, taus, sigma=1)

# ------------------------------- #
# Do Data Inversion without noise #
# ------------------------------- #
#'nl' = noiseless
#print('data inversions without noise')
#tmp = MEMBpy.MEMBryan_solve(A, b_clean/(A @ x)[0], C, m, alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=31, max_iter=10)
#x_pMEM_nl, x0_pMEM_var_nl, proposed_solutions_pMEM_nl, P_pMEM_nl, acceptance_arr_pMEM_nl = tmp
#tmp = MEMdNpy.MEMdN_solve(A, b_clean/(A @ x)[0], C, m,
#                          alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=31,
#                          rtol=1e-6, atol =1e-6, xrtol=1e-6, xatol=1e-6)
#x_dMEM_nl, x0_dMEM_var_nl, proposed_solutions_dMEM_nl, P_dMEM_nl, acceptance_arr_dMEM_nl = tmp

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

xmax=3.5
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
#    BdN1_deltax_norm_list.append( norm( proposed_solutions_pMEM[i] - proposed_solutions_dMEM[i]) )
#    BdN2_deltax_norm_list.append( norm( proposed_solutions_pMEM_nl[i] - proposed_solutions_dMEM_nl[i]) )
#    BdN3_deltax_norm_list.append( norm( proposed_solutions_pMEM[i] - proposed_solutions_dMEM_nl[i]) )
#    BdN4_deltax_norm_list.append( norm( proposed_solutions_pMEM_nl[i] - proposed_solutions_dMEM[i]) )

#    Bryan_deltax_norm_list.append( norm( proposed_solutions_pMEM[i] - proposed_solutions_pMEM_nl[i]) )
#    Bryan_RHS_norm_list.append(  deltab_norm*A_norm/C_iid_norm/P_pMEM[i,0] )
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
plt.ylim([0,6.0])
plt.tick_params(axis='both', labelsize=16)
print('imgs/double-gaussian_proposals-primal_noise'+noiselevel+'_priorflat.pdf')
plt.savefig('imgs/double-gaussian_proposals-primal_noise'+noiselevel+'_priorflat.pdf', bbox_inches='tight', format='pdf')

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
#    dNewton_deltax_norm_list.append(norm( proposed_solutions_dMEM[i] - proposed_solutions_dMEM_nl[i]))
#    dNewton_RHS_norm_list.append(deltab_norm*A_norm/C_iid_norm/P_dMEM[i,0])
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
plt.ylim([0,6.0])
plt.tick_params(axis='both', labelsize=16)
print('imgs/double-gaussian_proposals-dual_noise'+noiselevel+'_priorflat.pdf')
plt.savefig('imgs/double-gaussian_proposals-dual_noise'+noiselevel+'_priorflat.pdf', bbox_inches='tight', format='pdf')

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
plt.ylim([0,6.0])
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)
print('imgs/double-gaussian_reconstructions_noise'+noiselevel+'_prior-flat.pdf')
plt.savefig('imgs/double-gaussian_reconstructions_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')

# --------------------------- #
# PLOT POSTERIOR DISTRIBUTION #
# --------------------------- #
plt.figure()
j=0
for P, label, color, marker in zip([P_pMEM,P_dMEM],
                           [r'Bryan $P(\alpha| b, \mu )$', r'dual Newton $P(\alpha | b, \mu )$'],
                           ['blue', 'red'],
                           ['s','^']):
    prob = np.exp(np.sum(P[:,1:], axis=1) ) #/P_dMEM[start:,0]
    plt.plot(P[:,0], prob / np.sum(prob), color=color)
    plt.plot(P[j:,0], prob[j:] / np.sum(prob), label=label, color=color, marker=marker, ms=5, markevery=2)
    j+=1

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
print('imgs/double-gaussian_posteriorprob_noise'+noiselevel+'_prior-flat.pdf')
plt.savefig('imgs/double-gaussian_posteriorprob_noise'+noiselevel+'_prior-flat.pdf', bbox_inches='tight', format='pdf')
