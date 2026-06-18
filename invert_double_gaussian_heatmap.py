import time
import numpy as np; pi=np.pi
import src.C2KBryan as C2KBpy
#import src.C2KdualNewton as C2KdNpy
import matplotlib.pyplot as plt
from matplotlib import cm
colormap = cm.viridis  # You can change to 'plasma', 'jet', etc.
from matplotlib.colors import LogNorm

from numpy.linalg import norm, svd
from scipy.interpolate import RegularGridInterpolator

def interpolate_missing(data, method='linear', fill_value=None):
    # Interpolates NaN or zero entries in a 2D NumPy array using RegularGridInterpolator.
    #
    # Parameters:
    #   data (np.ndarray): 2D array containing numbers, NaNs, or zeros.
    #   method (str): Interpolation method: 'linear' or 'nearest'.
    #   fill_value (float or None): Value to use for points outside interpolation range (default: None).
    #
    # Returns:
    #   np.ndarray: Interpolated array with missing values filled.

    # Check input
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Replace zeros with NaN for interpolation
    data = np.where(data == 0, np.nan, data)

    # Define coordinate grid
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])

    # Mask of valid (non-NaN) data
    mask = ~np.isnan(data)

    # Stop if all values are NaN
    if not np.any(mask):
        raise ValueError("No valid data points found for interpolation.")

    # Get coordinates of valid points and their values
    points = np.array(np.nonzero(mask)).T
    values = data[mask]

    # Build interpolator on the full grid, temporarily filling NaNs with the mean
    interpolator = RegularGridInterpolator(
        (x, y),
        np.where(np.isnan(data), np.nanmean(data), data),
        method=method,
        bounds_error=False,
        fill_value=fill_value
    )

    # Create a full grid of coordinates
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    all_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Interpolate values across the full grid
    interpolated_values = interpolator(all_points).reshape(data.shape)

    # Replace NaN entries in original data with interpolated values
    filled_data = data.copy()
    filled_data[np.isnan(filled_data)] = interpolated_values[np.isnan(filled_data)]

    return filled_data

# ------------------------- #
# ---- Double Gaussian ---- #
# ------------------------- #
# This spectrum is text case 1 from Goulko, Olga, et al.
# "Numerical analytic continuation: Answers to well-posed questions."
# Physical Review B 95.1 (2017): 014102.
Nomega = 150;
omegas = np.linspace(4.0/Nomega, 4.0, Nomega, endpoint=True)  # units GeV
dw = omegas[1] - omegas[0] # units GeV
Ntau = 30;
taus = np.linspace(0, 5, Ntau);  # units GeV
dtau = taus[1] - taus[0];
print(Nomega, Ntau)
print('dw', dw, 'dtau', dtau)

c1=0.62; sigma1=0.12; z1=0.74
c2=0.41; sigma2=0.064; z2=2.93
x = c1/sigma1*np.exp(-(omegas-z1)**2/2/sigma1**2 )
x += c2/sigma2*np.exp(-(omegas-z2)**2/2/sigma2**2 )
flat = (np.sum(x)/len(x))*np.ones_like(x)

EE, TT = np.meshgrid(omegas, taus)
A = TT*EE
for i in range(len(taus)):
    for j in range(len(omegas)):
        A[i,j]= np.exp(-A[i,j])
b_clean = A @ x

# Declare Hyper parameters
noise_min = 1e-5
noise_max = 1e0
noise_vals= np.geomspace(noise_min, noise_max, 6, endpoint=True)
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<']
c_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) #np.linspace(0.05, 1.0, Nc, endpoint=True)
#noise_vals=np.array([1e-1, 1.0])
#c_vals = np.array([0.05, 0.1])
Nnoise = len(noise_vals)
Nc = len(c_vals)
Nsamples=100
Nruns=100

Nalpha=61
# dual Newton tolerance
rtol=1e-9
atol=1e-9

"""
dual_MSE = np.ones((Nnoise, Nc))
primal_MSE = np.ones((Nnoise, Nc))
dual_primal_deviation = np.ones((Nnoise, Nc))
dual_x = np.ones((Nnoise, Nc, Nomega))
primal_x = np.ones((Nnoise, Nc, Nomega))
alpha_tilde = np.ones((Nnoise, Nc))
start_time = time.time()
for _ in range(Nruns):
    print('run started')
    for i, noise_level in enumerate(noise_vals):
        if (noise_level <= 1e-5):
            alpha_min=1e-1
            alpha_max=1e22
        else:
            alpha_min=1e-3
            alpha_max=1e20

        stdev = noise_level * b_clean
        for j, c in enumerate(c_vals):
            #print('round (noise, c)', noise_level, c )
            b_samples = np.zeros((Nsamples,Ntau))
            for k in range(Nsamples):
                b_samples[k] = b_clean + np.random.normal(np.zeros((Ntau,)), stdev)
            b_avg = np.average(b_samples, axis=0)
            b_std = np.std(b_samples, axis=0, ddof=1)/np.sqrt(Nsamples) # std err
            b = np.reshape( b_avg/b_avg[0], (-1,1))
            C = np.diag( (b_std/b_avg[0])**2 )
            mu = (1-c)*x + c*flat
            m = np.reshape(mu/b_avg[0], (-1,1))
            #print('b_avg shape', np.shape(b_avg), 'b_avg[0]', b_avg[0], 'sum x', np.sum(x))
            #print('b[0]', b[0], 'sum mu', np.sum(m))

            tmp = C2KdNpy.C2KdN_solve(A, b, C, m,
                                      alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                                      rtol=rtol, atol=atol)
            x_dN, x_dN_var, proposed_solutions_dN, chi2_arr_dN, best_alphas = tmp
            #print('dual chi2', chi2_arr_dN[:,1])
            #print('dual best alphas', best_alphas)

            tmp = C2KBpy.C2KBryan_solve(A, b, C, m,
                                        alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha)
            x_B, x_B_var, proposed_solutions_B, chi2_arr_B, best_alphas = tmp
            #print('primal chi2', chi2_arr_B[:,1])
            #print('primal best alphas', best_alphas)

            # geometric mean
            dual_MSE[i,j] *= np.linalg.norm(b_avg[0]*x_dN - x)**(2/Nruns)
            primal_MSE[i,j] *= np.linalg.norm(b_avg[0]*x_B - x)**(2/Nruns)
            dual_primal_deviation[i,j] *= np.linalg.norm(b_avg[0]*(x_dN - x_B))**(2/Nruns)
            dual_x[i,j] *= (b_avg[0]*x_dN)**(1/Nruns)
            primal_x[i,j] *= (b_avg[0]*x_B)**(1/Nruns)
            geom_prod = best_alphas.prod()**(1.0/len(best_alphas))
            alpha_tilde[i,j] *= (noise_level**2*geom_prod)**(1/Nruns)

            # arithmatic mean
            #dual_MSE[i,j] += np.linalg.norm(b_avg[0]*x_dN - x)
            #primal_MSE[i,j] += np.linalg.norm(b_avg[0]*x_B - x)
            #dual_primal_deviation[i,j] += b_avg[0]*np.linalg.norm(x_dN - x_B)
            #dual_x[i,j] += b_avg[0]*x_dN
            #primal_x[i,j] += b_avg[0]*x_B
            #geom_prod = best_alphas.prod()**(1.0/len(best_alphas))
            #alpha_tilde[i,j] += noise_level**2*geom_prod


# arithmatic mean
#dual_MSE /= Nruns
#primal_MSE /= Nruns
#dual_primal_deviation /= Nruns
#dual_x /= Nruns
#primal_x /= Nruns
#alpha_tilde /= Nruns

np.savez('data/MSE_datafile_working.npz', dual_MSE=dual_MSE,
                                          primal_MSE=primal_MSE,
                                          dual_primal_deviation=dual_primal_deviation,
                                          dual_x=dual_x,
                                          primal_x=primal_x,
                                          alpha_tilde=alpha_tilde)

"""
tmp=np.load('data/MSE_datafile_working.npz')
dual_MSE=tmp['dual_MSE']
primal_MSE=tmp['primal_MSE']
dual_primal_deviation=tmp['dual_primal_deviation']
dual_x=tmp['dual_x']
primal_x=tmp['primal_x']
alpha_tilde=tmp['alpha_tilde']

print('dual MSE', dual_MSE)
print('primal MSE', primal_MSE)
print('alpha_tilde', alpha_tilde)

print('cvals', c_vals)
error_vals = noise_vals/np.sqrt(Nsamples)
for j, c in enumerate(c_vals):
    print('cvals', c)
    mu = (1-c)*x + c*flat
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4), gridspec_kw={'wspace': 0})
    fig.suptitle(f'Solutions across noise level at c={c:.2f}', fontsize=22)
    for ax in [ax1, ax2]:
        ax.plot(omegas, x, label='exact', c='black', marker='o', markevery=3)
        ax.plot(omegas, mu, label='prior',  c='black', marker='x', markevery=3)
        for i, noise in enumerate(error_vals):
            ax.plot(omegas, dual_x[i,j], label=r'err. '+f'{noise:.0e}', marker=markers[i], markevery=2*i+1)
            #plt.plot(omegas, primal_x[i,j], label=f'primal noise={noise:.1e}', marker='^', markevery=2*i+3)
        ax.set_xlabel(r'$\omega$ (GeV)', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
    ax1.set_xlim([0.25, 1.25])
    ax2.set_xlim([2.5, 3.5])
    if (c == 0.5):
        ax2.legend(fontsize=14, loc='upper right')
        #ax2.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1.0)) # outside the box
    #ax2.legend(fontsize=16, loc='upper right')
    plt.savefig('imgs/double-gaussian_c'+str(np.round(c,1))+'.pdf', bbox_inches='tight', format='pdf')
    print('saved figure to: imgs/double-gaussian_c'+str(np.round(c,1))+'.pdf')

# ------------------------ #
# Plot heat map of the MSE #
# ------------------------ #
fig, ax = plt.subplots(figsize=(8, 6))
MSE = dual_MSE
# Log colormap normalization and contour leves
vmin = np.nanmin(MSE[MSE > 0])
vmax = np.nanmax(MSE)
norm = LogNorm(vmin=vmin, vmax=vmax)
levels=[1,10,100]

# Create the pcolormesh
c_grid, noise_grid = np.meshgrid(c_vals, error_vals)
mesh = ax.pcolormesh(c_grid, noise_grid, MSE,
                         cmap='viridis', shading='gouraud', norm=norm)

# Add red x's at grid points
for i, error_level in enumerate(error_vals):
    for j, c in enumerate(c_vals):
        ax.scatter(c, error_level, marker='x', color='r')

# Add contour lines
cs = ax.contour(c_vals, error_vals, MSE, levels=levels, norm=norm, colors='black')

# Colorbar
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Mean Squared Error (MSE)', fontsize=18)
# add black lines of contours into heat map
for lev in levels:
    cbar.ax.hlines(lev, 0, 1, colors='black', linewidth=1.2)

# title and axis
ax.set_title('MSE of Max Ent. via dual Newton optimizer', fontsize=20, pad=13)

ax.set_xlabel('c (varies prior)', fontsize=18)
ax.tick_params(axis='x', labelsize=16)

ax.set_ylabel(r'Data error $\sigma_0/N_S^{1/2}$ (varies noise)', fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.set_yscale('log')
ax.set_ylim([np.amin(error_vals), np.amax(error_vals)])


plt.tight_layout()
plt.savefig('dual_MSE_heatmap.pdf', format='pdf')
print('saved figure: dual_MSE_heatmap.pdf')

# ------------------------------------------- #
# Plot heat map of difference between methods #
# ------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 6))
dual_x_norm = np.zeros((Nnoise, Nc))
for i, error_level in enumerate(error_vals):
    for j, c in enumerate(c_vals):
        dual_x_norm[i,j] = np.linalg.norm(dual_x[i,j])**2

print('dual_x_norm', dual_x_norm)
dual_x_norm[dual_x_norm < 1e-9] = 0.0
print('dual_x_norm', dual_x_norm)
dual_x_norm = interpolate_missing(dual_x_norm, method='cubic')
print('dual_x_norm', dual_x_norm)

Z = dual_primal_deviation / dual_x_norm

# Log colormap normalization and contour leves
vmin = np.nanmin(Z[Z > 0])
vmax = np.nanmax(Z)
norm = LogNorm(vmin=vmin, vmax=vmax)
levels=[1,10,100]

# Create the pcolormesh
c_grid, noise_grid = np.meshgrid(c_vals, error_vals)
mesh = ax.pcolormesh(c_grid, noise_grid, Z,
                         cmap='viridis', shading='gouraud', norm=norm)

# Add red x's at grid points
for i, error_level in enumerate(error_vals):
    for j, c in enumerate(c_vals):
        ax.scatter(c, error_level, marker='x', color='r')

# Add contour lines
#cs = ax.contour(c_vals, noise_vals, MSE, levels=levels, norm=norm)

# Colorbar
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(r'Average $||x_\mathrm{dual N.} - x_\mathrm{Bryan}|| \, / \, ||x_\mathrm{dual N.} ||$', fontsize=18)
# add black lines of contours into heat map
for lev in levels:
    cbar.ax.hlines(lev, 0, 1, colors='black', linewidth=1.2)

# title and axis
ax.set_title('Avg. relative difference between estimates', fontsize=20, pad=13)

ax.set_xlabel('c (varies prior)', fontsize=18)
ax.tick_params(axis='x', labelsize=16)

ax.set_ylabel(r'Data error $\sigma_0/N_S^{1/2}$ (varies noise)', fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.set_yscale('log')
ax.set_ylim([np.amin(error_vals), np.amax(error_vals)])


plt.tight_layout()
plt.savefig('dual_primal_diff.pdf', format='pdf')
print('saved figure: dual_primal_diff.pdf')

