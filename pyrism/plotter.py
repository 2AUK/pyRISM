import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

df = pd.read_csv(sys.argv[1], sep=',', skiprows=[0])
r = df.iloc[:, 0].to_numpy()
gr = df.iloc[:, int(sys.argv[2])].to_numpy()
r = np.asarray(r, dtype=np.float64)
gr = np.asarray(gr, dtype=np.float64)

plt.axhline(1, color='grey', linestyle="--", linewidth=2)
plt.xlabel("r/A")
plt.ylabel("g(r)")
plt.plot(r, gr, color='black')
#plt.xlim([2, 10])
plt.savefig(sys.argv[1] + '_RDF.eps', format='eps')
plt.show()
