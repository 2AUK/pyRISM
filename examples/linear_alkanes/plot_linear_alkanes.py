from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

inps = Path('.').glob('*.td')

x = []
y = []
for i in inps:
    with open(i, 'r') as tdfile:
        for line in tdfile:
            line = line.split()
            print(line)
            if line[0] == "RISM":
                x.append(float(line[-2]))
            elif line[0] == "PW:":
                y.append(float(line[-1]))


x = np.asarray(x)
y = np.asarray(y)
table = list(zip(x, y))
print(tabulate(table, headers=["PMV", "PW SFE"]))

plt.plot(x, y, 'k.')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

