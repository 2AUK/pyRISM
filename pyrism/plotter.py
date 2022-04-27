import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt

df = pd.read_csv(sys.argv[1], sep=',', skiprows=[0])
r = df.iloc[:, 0].to_numpy()
grs = []
gr_peaks = []
print(sys.argv[2])
print(df.shape)
if sys.argv[2] == "-":
    for i in range(1, df.shape[1]):
        gr = np.asarray(df.iloc[:, i].to_numpy())
        grs.append((df.columns[i], gr))
        gr_peaks.append(find_peaks(gr, height=1))
else:
    for i in sys.argv[2:]:
        i = int(i)
        gr = np.asarray(df.iloc[:, i].to_numpy())
        grs.append((df.columns[i], np.asarray(df.iloc[:, i].to_numpy())))
        gr_peaks.append(find_peaks(gr, height=1))
r = np.asarray(r, dtype=np.float64)

plt.axhline(0, color='grey', linestyle="--", linewidth=2)
plt.xlabel("r/A")
plt.ylabel("g(r)")

for i, gr in enumerate(grs):
    indices = gr_peaks[i][0]
    plt.plot(r, gr[1], label="pyRISM " + gr[0])
    plt.plot(r[indices], gr[1][indices], 'x')

plt.legend()
plt.savefig(sys.argv[1] + '_RDF.eps', format='eps')
plt.show()
