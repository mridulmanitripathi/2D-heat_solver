#!/usr/bin/env bash
# Lecture queue
#SBATCH --account=lect0045
# Outputs of the job
#SBATCH --output=out.%j
#SBATCH --error=err.%j
# Wall clock limit
#SBATCH --time=0:51:00

# run the process
../build/2d_Unsteady ./settings.finest.in
