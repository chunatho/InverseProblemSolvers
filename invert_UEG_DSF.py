import numpy as np; pi = np.pi
print('np version', np.version.version)
from scipy.integrate import quad
from scipy.interpolate import interp1d
from src.PhysicalConstantsUnitConversions import *

import src.C2KBryan as C2KBpy
import src.C2KdualNewton as C2KdNpy
import sys
import time
import matplotlib.pyplot as plt


def find_omega_cut(arr, cut, direction_from_peak):
    max_index = np.argmax(arr)
    if direction_from_peak == 'positive':
        for i in range(max_index, len(arr)):
            if arr[i] < cut:
                return i
        return len(arr) - 1
    elif direction_from_peak == 'negative':
        for i in range(0, max_index):
            if arr[i] > cut:
                return i
        return 0
    else:
        print('Invalid direction')
        sys.exit(1)


def main():
    rs = 20
    Theta = 1
    kindex = 15

    model_file = f'data/UEG_DSF_data_prior_matrix/UEG_density_response_kindex{kindex}_rs{rs}_theta{Theta}_prior.dat'
    F_file_list = [
        f'data/UEG_DSF_data_prior_matrix/UEG_density_response_kindex{kindex}_rs{rs}_theta{Theta}_samples10.txt',
        f'data/UEG_DSF_data_prior_matrix/UEG_density_response_kindex{kindex}_rs{rs}_theta{Theta}_samples50.txt',
        f'data/UEG_DSF_data_prior_matrix/UEG_density_response_kindex{kindex}_rs{rs}_theta{Theta}_samples100.txt',
        f'data/UEG_DSF_data_prior_matrix/UEG_density_response_kindex{kindex}_rs{rs}_theta{Theta}_samples10000.txt'
    ]

    file_labels = [r'$N_s=10$', r'$N_s = 50$', r'$N_s = 10^{2}$', r'$N_s = 10^{4}$']

    N_files = len(F_file_list)

    Nalpha = 30
    alpha_min = 1e-6
    alpha_max = 1e13
    C2K_min = 2
    C2K_max = 2.5
    NC2K = 10
    c_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    # ---- Physics ----
    rs_cm = rs * aB_cm
    n_e = 1.0 / (4./3. * pi * rs_cm**3)
    k_Fe_invcm = (3 * pi * pi * n_e)**(1/3)
    E_Fe_erg = (hbar_ergs * k_Fe_invcm)**2 / 2 / me_g
    T_erg = Theta * E_Fe_erg
    beta_invHartree = 1.0 / T_erg / erg_to_Hartree

    # ---- Load prior ----
    model_data = np.loadtxt(model_file, delimiter=',', skiprows=1)
    ws_Hartree = model_data[:, 0]
    S_static = model_data[:, 1]

    wcut_index_a = find_omega_cut(S_static, 1e-8*np.max(S_static), 'negative')
    wcut_index_b = find_omega_cut(S_static, 1e-8*np.max(S_static), 'positive')

    ws_reduced = ws_Hartree[wcut_index_a:wcut_index_b]
    dw_Hartree = ws_Hartree[1] - ws_Hartree[0]
    N_omega = wcut_index_b - wcut_index_a
    print('omega interval', ws_reduced[0], ws_reduced[-1], 'omega spacing', dw_Hartree, 'N_omega', N_omega)

    # Flat prior
    FLAT = np.ones(N_omega)
    norm = np.sum(S_static[wcut_index_a:wcut_index_b])
    FLAT = FLAT * norm / np.sum(FLAT)

    # ---- Storage ----
    Bryan_sols = {}
    Bryan_vars = {}

    dual_N_sols = {}
    dual_N_vars = {}
    noise_list = []
    # =========================
    # Loop over files
    # =========================
    for file_idx, f_file in enumerate(F_file_list):
        print(f'\nProcessing: {f_file}')

        data = np.loadtxt(f_file, delimiter=',')
        taus = data[:, 0]
        F = data[:, 1].reshape(-1, 1)
        dF = data[:, 2].reshape(-1, 1)

        Ntau = len(taus)

        # Save Laplace Transform for MPF
        beta_tau = taus[-1]
        EE, TT = np.meshgrid(ws_Hartree, taus)
        A = np.exp(-TT * EE) + np.exp(-(beta_tau - TT) * EE)
        np.savetxt(f'/home/chuna69/InverseProblemSolvers/data/UEG_DSF_data_prior_matrix/UEG_density_response_rs{rs}_theta{Theta}_matrix.dat', A, delimiter=',')

        sys.exit(1)

        beta_tau = taus[-1]
        EE, TT = np.meshgrid(ws_reduced, taus)
        A = np.exp(-TT * EE) + np.exp(-(beta_tau - TT) * EE)

        F0 = float(F[0,0])
        b = F/F0
        b_err = dF[:, 0]/F0
        noise_list.append(np.average(b_err/b))
        C_iid = np.diag(b_err**2)

        normalization = dw_Hartree / F0

        for c in c_values:
            if(c==0.0 and file_idx < len(F_file_list)-1):
                continue # only run c=0.0 when for last file
            print(f'running c={c} on file {f_file}')
            mu_phys = c*FLAT + (1 - c)*S_static[wcut_index_a:wcut_index_b]
            mu = normalization*mu_phys.reshape(-1, 1)

            # ---- Bryan ----
            Skw, Skw_var, *_ = C2KBpy.C2KBryan_solve(
                A, b, C=C_iid, mu=mu,
                alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha,
                C2K_min=C2K_min, C2K_max=C2K_max, NC2K=NC2K
            )

            Bryan_sols[(file_idx, c)] = Skw.flatten() / normalization
            Bryan_vars[(file_idx, c)] = Skw_var.flatten() / (normalization**2)

            # ---- Dual Newton ----
            #Skw, Skw_var, *_ = C2KdNpy.C2KdN_solve(
            #    A, b, C=C_iid, mu=mu,
            #    alpha_min=alpha_min, alpha_max=alpha_max, Nalpha=Nalpha
            #)

            dual_N_sols[(file_idx, c)] = Skw.flatten() / normalization
            dual_N_vars[(file_idx, c)] = Skw_var.flatten() / (normalization**2)


    for key, value in Bryan_sols.items():
        print('Bryan keys', key)
    for key, value in dual_N_sols.items():
        print('dual N keys', key)
    # =========================
    # Plotting
    # =========================
    colors = ['r', 'g', 'b', 'purple'] #plt.cm.viridis(np.linspace(0.1, 0.9, N_files))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<']

    methods = [
        ('Dual Newton', dual_N_sols, dual_N_vars, 'dualNewton'),
        ('Bryan', Bryan_sols, Bryan_vars, 'Bryan')
    ]

    for method_label, sol_dict, var_dict, tag in methods:
        #S_exact = sol_dict[(len(F_file_list)-1, 0.0)] #S_static[wcut_index_a:wcut_index_b]
        S_exact = S_static[wcut_index_a:wcut_index_b]
        for c in c_values[1:]:
            print(f'plotting c={c}')
            mu_plot = c * FLAT + (1 - c) * S_exact

            fig, ax = plt.subplots(figsize=(8, 5))

            #ax.plot(ws_reduced, S_exact, color='black', lw=2, label=f'exact (Nsamples $10^{4}$, c=0.0)')
            ax.plot(ws_reduced, S_exact, color='black', lw=2, label=f'exact (static approx)')
            ax.plot(ws_reduced, mu_plot, '--', color='gray', label='prior')

            for file_idx, f_file in enumerate(F_file_list):
                y = sol_dict[(file_idx, c)]
                var = var_dict[(file_idx, c)]
                yerr = 2.0 * np.sqrt(np.abs(var))
                color = colors[file_idx]
                marker = markers[file_idx]

                # Main curve (fully opaque)
                ax.plot(
                    ws_reduced, y,
                    color=color,
                    marker=marker,
                    markevery = 29 + 3*file_idx,
                    lw=1.2,
                    label=file_labels[file_idx] + r' err. ' + f'{noise_list[file_idx]:.1e}'
                )

                # Shaded uncertainty band (transparent)
                ax.fill_between(
                    ws_reduced,
                    y - yerr,
                    y + yerr,
                    color=color,
                    alpha=0.2,   # controls opacity of the band
                    linewidth=0  # no border line
                )
            ax.set_title(f'Solutions across noise level at $c={c}$', fontsize=20)
            ax.set_xlabel(r'$\omega$ [Hartree]', fontsize=18)
            ax.set_ylabel(r'$S(q, \omega)$ for $q = 3.445$ 1/Bohr', fontsize=18)

            ax.set_ylim([0, 1.2*np.amax(S_exact)])

            if (c == 0.25):
                ax.legend(fontsize=14)
            fig.tight_layout()
            plt.tick_params(axis='both', labelsize=16)

            outname = f'UEGDSF_rs{rs}_Theta{Theta}_c{c}_{tag}.png'
            fig.savefig('imgs/'+outname, dpi=150, bbox_inches='tight')
            print(f'Saved: {outname}')
            plt.close(fig)


if __name__ == "__main__":
    main()
