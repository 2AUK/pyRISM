import sys
import os
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
full_data_dir = Path(sys.argv[1]).resolve()
data_dir = Path(sys.argv[2]).resolve()

names = []
final_names = []
final_solvent = []
final_solver = []
final_times = []
final_iterations = []

solvers = ["ADIIS", "MDIIS", "Picard", "Ng"]
solvents = ["acetonitrile_outputs", "cSPCE_outputs", "cSPCE_NaCl_outputs", "chloroform_outputs"]

for toml in full_data_dir.rglob("*.toml"):
    names.append(toml.stem)

for solver in solvers:
    for solvent in solvents:
        for name in names:
            diagnostic_file = (data_dir / Path(solver) / Path(solvent) / Path(name + ".diag"))
            time = 1e20
            iterations = 1e20
            final_names.append(name)
            solvent_tok = solvent.split("_")
            if len(solvent_tok) == 3:
                final_solvent.append(solvent_tok[0] + "_" + solvent_tok[1])
            else:
                final_solvent.append(solvent_tok[0])
            if diagnostic_file.is_file():
                with open(diagnostic_file, 'r') as diag_file:
                    for line in diag_file.readlines():
                        line = line.split()
                        if line[0] == "Solute-Solvent":
                            if line[2] == "Time:":
                                time = float(line[-2])
                            if line[2] == "Iterations:":
                                iterations = int(line[-1])
                final_solver.append(solver)
                final_times.append(time)
                final_iterations.append(iterations)
            else:
                final_solver.append(solver)
                final_times.append(time)
                final_iterations.append(iterations)

performance_dict = {
        "Solute": final_names,
        "Solvent": final_solvent,
        "Solver": final_solver,
        "t": final_times,
        "i": final_iterations,
        }

df = pl.from_dict(performance_dict)
df = df.sort("Solver").sort("Solvent").sort("Solute")
print(df)
df_ratio_t = df.groupby(["Solute", "Solvent"]).agg(
        (pl.col("t") / pl.min("t")).alias('r_t')
        ).sort("Solute").sort("Solvent")
df_ratio_i = df.groupby(["Solute", "Solvent"]).agg(
        (pl.col("i") / pl.min("i")).alias('r_i')
        ).sort("Solute").sort("Solvent")
print(df_ratio_t)
print(df_ratio_i)

num_p = 740
tau_space = np.linspace(0, 10, num=11)
print(tau_space)
rho_tau_adiis = []
rho_tau_mdiis = []
rho_tau_picard = []
rho_tau_ng = []

for tau in tau_space:
    size_p_adiis = 0
    size_p_mdiis = 0
    size_p_picard = 0
    size_p_ng= 0
    for elem in df_ratio_t['r_t']:
        if elem[0] <= tau:
            size_p_adiis += 1
        if elem[1] <= tau:
            size_p_mdiis += 1
        if elem[2] <= tau:
            size_p_ng += 1
        if elem[3] <= tau:
            size_p_picard += 1
    rho_tau_adiis.append(size_p_adiis / num_p)
    rho_tau_mdiis.append(size_p_mdiis / num_p)
    rho_tau_picard.append(size_p_picard / num_p)
    rho_tau_ng.append(size_p_ng / num_p)

plt.step(tau_space, rho_tau_picard, label='Picard')
plt.step(tau_space, rho_tau_ng, label='Ng')
plt.step(tau_space,rho_tau_mdiis, label='MDIIS2')
plt.step(tau_space, rho_tau_adiis, label='MDIIS')
plt.xlabel('\tau')
plt.ylabel('\rho_s(\tau)')
plt.legend()
plt.show()
plt.savefig("performance_profile_iter.png")
