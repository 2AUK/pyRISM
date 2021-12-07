import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

df = pd.read_csv(sys.argv[1], sep=',', skiprows=[0])
r = df.iloc[:, 0].to_numpy()
grs = []
for i in sys.argv[2:]:
    i = int(i)
    grs.append((df.columns[i], np.asarray(df.iloc[:, i].to_numpy())))
r = np.asarray(r, dtype=np.float64)

plt.axhline(1, color='grey', linestyle="--", linewidth=2)
plt.xlabel("r/A")
plt.ylabel("g(r)")

for gr in grs:
    plt.plot(r, gr[1], label="pyRISM " + gr[0])

plt.legend()
plt.savefig(sys.argv[1] + '_RDF.eps', format='eps')
plt.show()
