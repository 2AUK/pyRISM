import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
df = pd.read_csv(sys.argv[1], sep=',', skiprows=[0])
print(df)
df = df.sort_values(by=['epsilon'], ascending=False)
df['idx'] = df['idx'].sort_values(ascending=True).values
print(df)
r = df.iloc[:, 0].to_numpy()
grs = []
print(sys.argv[2])
print(df.shape)
if sys.argv[2] == "-":
    for i in range(1, df.shape[1]):
        grs.append((df.columns[i], np.asarray(df.iloc[:, i].to_numpy())))
else:
    for i in sys.argv[2:]:
        i = int(i)
        grs.append((df.columns[i], np.asarray(df.iloc[:, i].to_numpy())))
r = np.asarray(r, dtype=np.float64)

plt.axhline(1, color='grey', linestyle="--", linewidth=2)
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$t_{\textit{Picard}}/t_{\textit{method}}$')
plt.xlim([0,12])
ticks = [1e-4, 5e-5, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8, 5e-8, 1e-9, 5e-9, 1e-10]
ticks.sort(reverse=True)
print(ticks)
plt.gca().xaxis.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
plt.gca().xaxis.set_ticklabels(ticks)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
for gr in grs:
    plt.plot(r, gr[1], label=gr[0])

plt.legend()
plt.savefig(sys.argv[1] + '_RDF.eps', format='eps')
plt.show()
