import sys
import os
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

original_data_dir = Path(sys.argv[1]).resolve()
diagnostic_data_dir = Path(sys.argv[2]).resolve()


names = []
succeeded_names = []
v_size = []
u_size = []
size = []
vv_times = []
uv_times = []
times = []
vv_iterations = []
uv_iterations = []
iterations = []
for toml in original_data_dir.rglob("*.toml"):
    names.append(toml.stem)

for diag in diagnostic_data_dir.rglob("*.diag"):
    name = diag.stem
    succeeded_names.append(name)
    with open(diag, "r") as diagfile:
        for line in diagfile.readlines():
            line = line.split()
            print(line)
            if line[2] == "Size:":
                if line[0] == "Solvent":
                    v_size.append(int(line[-1]))
                if line[0] == "Solute":
                    u_size.append(int(line[-1]))
                if line[0] == "Total":
                    size.append(int(line[-1]))
            elif line[0] == "Solvent-Solvent":
                if line[2] == "Time:":
                    vv_times.append(float(line[-2]))
                if line[2] == "Iterations:":
                    vv_iterations.append(int(line[-1]))
            elif line[0] == "Solute-Solvent":
                if line[2] == "Time:":
                    uv_times.append(float(line[-2]))
                if line[2] == "Iterations:":
                    uv_iterations.append(int(line[-1]))
            elif line[0] == "Total" and line[2] == "Time:":
                times.append(float(line[-2]))
        iterations.append(vv_iterations[-1] + uv_iterations[-1])

differences = set(names) - set(succeeded_names)
differences = list(differences)

performance_dict = {
    "Jobs": succeeded_names,
    "Solvent Size": v_size,
    "Solute Size": u_size,
    "Total Size": size,
    "Solvent-Solvent Time (s)": vv_times,
    "Solvent-Solvent Iterations": vv_iterations,
    "Solute-Solvent Time (s)": uv_times,
    "Solute-Solvent Iterations": uv_iterations,
    "Total Job Time (s)": times,
    "Total Iterations": iterations,
}
df = pl.from_dict(performance_dict)

df_original = df.clone()
df = df.groupby("Total Size").mean()
df_original = df_original.groupby("Total Size").std()

df = df.sort("Total Size").with_columns(
    (pl.col("Total Job Time (s)") / pl.min("Total Job Time (s)")).alias(
        "Normalised Total Time"
    ),
    (pl.col("Total Iterations") / pl.min("Total Iterations")).alias(
        "Normalised Iterations"
    ),
)
df = df.sort("Total Size")

df = df.filter(pl.col("Total Size") < 40)
print(df_original)

plt.plot(
    df.select(pl.col("Total Size")).to_numpy(),
    df.select(pl.col("Normalised Total Time")).to_numpy(),
    marker="o",
    linestyle="none",
)
plt.show()
