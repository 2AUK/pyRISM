import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np


def plot():
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = 14
    df = pd.read_csv(sys.argv[1], sep=",", skiprows=[0])

    r = df.iloc[:, 0].to_numpy()
    grs = []
    if sys.argv[2] == "-":
        for i in range(1, df.shape[1]):
            grs.append((df.columns[i], np.asarray(df.iloc[:, i].to_numpy())))
    else:
        for i in sys.argv[2:]:
            i = int(i)
            grs.append((df.columns[i], np.asarray(df.iloc[:, i].to_numpy())))
    r = np.asarray(r, dtype=np.float64)

    plt.axhline(1, color="grey", linestyle="--", linewidth=2)
    plt.xlabel(r"r (Angstrom)")
    plt.ylabel(r"g(r)")
    plt.xlim([0, 10])
    for gr in grs:
        plt.plot(r, gr[1], label=gr[0])

    plt.legend()
    plt.savefig(sys.argv[1] + "_RDF.eps", format="eps")
    plt.show()
