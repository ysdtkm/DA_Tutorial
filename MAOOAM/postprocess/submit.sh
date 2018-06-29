#!/bin/bash
#SBATCH -n 1
#SBATCH -t 04:15:00
#SBATCH -J "anl_clim_cov"

set -e
cd /lustre/tyoshida/prgm/DA_Tutorial/MAOOAM/postprocess
python3 analyze_long_run.py

