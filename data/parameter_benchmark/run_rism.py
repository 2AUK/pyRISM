#!/usr/bin/env python
import subprocess
import tempfile
import sys
from pathlib import Path
import os
import numpy as np
import toml

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

# Parameter range for depth
depth_grid = list(np.arange(3, 21))
# Parameter range for damping
damping_grid = list(np.round(np.arange(0.1, 1.1, 0.1), 1))
# Parameter range for tolerances
tolerance_grid = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
# Parameter range for grid size
npts_grid = [2 ** x for x in range(7, 15)]

# Input toml file for parameter testing
in_toml = Path(sys.argv[1]).resolve()
# Output directory (new directories will be generated based on the parameter search)
out_dir = Path(sys.argv[2]).resolve()


if __name__ == "__main__":

    # Load the test problem
    prblm = toml.load(in_toml)

    # # Fix the parameters we're not changing
    # # 1024 grid points with a 0.02 grid spacing
    # # Hopefully at this resolution, the calculations are fast while still being able to probe the
    # # parameter space properly.
    # prblm["system"]["npts"] = 8192
    # prblm["system"]["radius"] = 40.96
    #
    # # Standard tolerance for most numerical methods
    # prblm["params"]["tol"] = 1e-8
    #
    # # Setting solver to MDIIS
    # prblm["params"]["solver"] = "ADIIS"
    #
    # # New directory for MDIIS depth and damping parameter search
    # mdiis_dir = out_dir / Path("mdiis_depth_damp")
    # mdiis_dir.mkdir(exist_ok=True)
    #
    # #Depth and damping probing for MDIIS
    # for depth in depth_grid:
    #     for damping in damping_grid:
    #         # Get names
    #         name = in_toml.stem
    #         fname = in_toml.name
    #
    #         # Get name for new folder
    #         descriptor = "{name}_ADIIS_de_{de}_da_{da}".format(name=name, de=depth, da=damping)
    #
    #         # Generate directory for the specific problem
    #         new_dir = mdiis_dir / Path(descriptor)
    #         new_dir.mkdir(exist_ok=True)
    #
    #         # cd into new directory
    #         os.chdir(new_dir)
    #
    #         # Modify problem
    #         prblm["params"]["mdiis_damping"] = damping
    #         prblm["params"]["depth"] = depth
    #
    #         # Write problem to directory
    #         with open(fname, 'w') as toml_file:
    #             toml.dump(prblm, toml_file, encoder=toml.TomlNumpyEncoder())
    #
    #         # Remove any previous SLURM job files
    #         for slurm in new_dir.glob("slurm*"):
    #             os.remove(slurm)
    #
    #         # Generate temporary submit script
    #         fp = tempfile.NamedTemporaryFile(dir=new_dir, suffix='.sh', buffering=0)
    #         sbatch = temp_sbatch.format(job=descriptor, toml_file=fname)
    #         fp.write(str.encode(sbatch))
    #
    #         # Submit job
    #         subprocess.run(["sbatch", fp.name])
    #
    #         # Close file handler; delete file
    #         fp.close()

    # Fix depth and damping
    prblm["params"]["mdiis_damping"] = 0.7
    prblm["params"]["depth"] = 20

    dr = 0.002
    # New directory for grid problem
    grid_dir = out_dir / Path("grid_spacing_{dr}".format(dr=str(dr)))
    grid_dir.mkdir(exist_ok=True)
    for npt in npts_grid:
        print(npt, npt * dr)
        # Get names
        name = in_toml.stem
        fname = in_toml.name

        # Get name for new folder
        descriptor = "{name}_ADIIS_grid_{npts}".format(name=name, npts=npt)
        # Generate directory for the specific problem
        new_dir = grid_dir / Path(descriptor)
        new_dir.mkdir(exist_ok=True)

        # cd into new directory
        os.chdir(new_dir)

        # Modify problem
        prblm["system"]["npts"] = npt
        prblm["system"]["radius"] = npt * dr

        # Write problem to directory
        with open(fname, "w") as toml_file:
            toml.dump(prblm, toml_file, encoder=toml.TomlNumpyEncoder())

        # Remove any previous SLURM job files
        for slurm in new_dir.glob("slurm*"):
            os.remove(slurm)

        # Generate temporary submit script
        fp = tempfile.NamedTemporaryFile(dir=new_dir, suffix=".sh", buffering=0)
        sbatch = temp_sbatch.format(job=descriptor, toml_file=fname)
        fp.write(str.encode(sbatch))

        # Submit job
        subprocess.run(["sbatch", fp.name])

        # Close file handler; delete file
        fp.close()
