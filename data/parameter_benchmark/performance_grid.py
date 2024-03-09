import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

inp_dir = Path(sys.argv[1]).resolve()

grid = []
vv_time = []
vv_iterations = []
uv_time = []
uv_iterations = []

for diag in inp_dir.rglob("*.diag"):
    grid_val = diag.parts[-2].split("_")[-1]
    print(grid_val)
    grid.append(float(grid_val))
    with open(diag, "r") as diag_file:
        for line in diag_file.readlines():
            line = line.split()
            if line[0] == "Solvent-Solvent":
                if line[2] == "Time":
                    vv_time.append(float(line[-2]))
                if line[2] == "Iterations":
                    vv_iterations.append(int(line[-1]))
            if line[0] == "Solute-Solvent":
                if line[2] == "Time":
                    uv_time.append(float(line[-2]))
                if line[2] == "Iterations":
                    uv_iterations.append(int(line[-1]))


plt.scatter(vv_time, grid)
plt.show()
