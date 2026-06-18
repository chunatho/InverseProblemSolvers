#!/bin/bash
#SBATCH --job-name=run_all_inversions
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/run_all_inversions.log
#SBATCH --ntasks=1 # Total number of tasks (MPI processes)
#SBATCH --cpus-per-task=1 # this indicates the max numbers of threads that prange can use.

#module load python/3.12.4 # rosi
module load python/3.12.2 # hemera
module load gcc/14.2.0

source .venv/bin/activate

#python invert_double_gaussian_heatmap.py

#python run_UEG_DSF.py

python invert_double_gaussian.py > logs/invert_double_gaussian.out
mv imgs/dictionarylearning_MSE.pdf imgs/doublegaussian_dictionarylearning_MSE_noise0p001.pdf

python invert_rhomeson.py > logs/invert_rhomeson.out
mv imgs/dictionarylearning_MSE.pdf imgs/rho-meson_dictionarylearning_MSE_noise0p001_prior-flat.pdf

python invert_gaussian_kernel.py > logs/invert_gaussian_kernel.out
mv imgs/dictionarylearning_MSE.pdf imgs/gaussianInversion_dictionarylearning_MSE_noise0p001.pdf

python invert_skew_gaussian.py > logs/skew_gaussian_skew-5.out
mv imgs/dictionarylearning_MSE.pdf imgs/skewgaussian_dictionarylearning_MSE_noise0p001_skew-5.pdf

