#!/bin/bash

module load python/3.12.2
module load gcc/14.2.0

source .venv/bin/activate
python invert_gaussian_kernel.py
python invert_rhomeson.py
python invert_double_gaussian.py
