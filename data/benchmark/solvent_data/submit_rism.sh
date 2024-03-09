#!/bin/bash

#======================================================
#
# Job script for running a serial job on a single core 
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=standard
#
# Specify project account
#SBATCH --account=palmer-mmms
#
# No. of tasks required (ntasks=1 for a single-core job)
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=40
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=168:00:00
#
# Job name
#SBATCH --job-name=test
#
# Output file
#SBATCH --output=test-%j.out
#======================================================

module purge

#example module load command (foss 2018a contains the gcc 6.4.0 toolchain & openmpi 2.12)   
module load openblas/gcc-8.5.0 
module load fftw/gcc-8.5.0

#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Modify the line below to run your program

rism acetonitrile.toml -l -c

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
