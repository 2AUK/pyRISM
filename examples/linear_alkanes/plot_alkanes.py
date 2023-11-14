import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ratkova_dimensionless_pmv = np.asarray([1.79, 2.45, 3.00, 3.47, 4.04, 4.54, 5.04, 5.57, 6.05, 6.57])
pyrism_dimensionless_pmv = []

p = Path('.')

inputs = list(p.glob("*.td"))

for i in inputs:
    with open(i, 'r') as tdfile:
        for line in tdfile:
            line = line.split()
            if line[0] == "RISM":
                pyrism_dimensionless_pmv.append(float(line[-2]))
                
pyrism_dimensionless_pmv = np.asarray(sorted(pyrism_dimensionless_pmv))
pyrism_dimensionless_pmv *= density
a, b = np.polyfit(ratkova_dimensionless_pmv, pyrism_dimensionless_pmv, 1)
_, ax = plt.subplots()
ax.axline((0.0, 0.0), (2.5, 2.5), color='k', linestyle='--')
ax.axline((ratkova_dimensionless_pmv[0], pyrism_dimensionless_pmv[0]), slope=a, color='orange')
plt.plot(ratkova_dimensionless_pmv, pyrism_dimensionless_pmv, 'r.')
plt.show()
