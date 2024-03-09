import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

inp_dir = Path(sys.argv[1]).resolve()


depth = []
damping = []
vv_time = []
vv_iterations = []
uv_time = []
uv_iterations = []

for diag in inp_dir.rglob("*.diag"):
    depth_val, damp_val = diag.parts[-2].split("_")[-3], diag.parts[-2].split("_")[-1]
    print(depth_val, damp_val)
    depth.append(int(depth_val))
    damping.append(float(damp_val))
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

perf_dict = {
        "depth": depth,
        "damp": damping,
        "vv_t": vv_time,
        "vv_i": vv_iterations,
        "uv_t": uv_time,
        "uv_i": uv_iterations,
        }

df = pl.from_dict(perf_dict)
df_depth = df.filter(pl.col("damp") == 0.5).sort("depth")
df_damp = df.filter(pl.col("depth") == 4).sort("damp")
print(df)
print(df_depth)
print(df_damp)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")


# ax.scatter(vv_time, damping, depth)
ax.scatter(uv_iterations, damping, depth)
ax.set_xlabel("iterations")
ax.set_ylabel("damping")
ax.set_zlabel("depth")
plt.show()

plt.scatter(df_depth["depth"], df_depth["vv_i"])
plt.show()

plt.scatter(df_damp["damp"], df_damp["vv_i"])
plt.show()
