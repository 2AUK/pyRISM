#!/usr/bin/env python
import subprocess
import tempfile
import sys
from pathlib import Path
import os

temp_sbatch = """#!/bin/bash

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
#SBATCH --cpus-per-task=4
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=168:00:00
#
# Job name
#SBATCH --job-name={job}_pyrism
#
# Output file
#SBATCH --output=slurm-{job}-%j.out
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

rism {toml_file} -l

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------"""


in_dir = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
Path.mkdir(out_dir, exist_ok=True)
if __name__ == "__main__":
    os.chdir(out_dir)
    for path in in_dir.rglob("*.toml"):
        print(path)
        for slurm in out_dir.rglob("slurm*"):
            os.remove(slurm)
        fp = tempfile.NamedTemporaryFile(dir=in_dir, suffix=".sh", buffering=0)
        sbatch = temp_sbatch.format(job=path.stem, toml_file=path)
        fp.write(str.encode(sbatch))
        print(fp.name)
        subprocess.run(["sbatch", fp.name])
        fp.close()
