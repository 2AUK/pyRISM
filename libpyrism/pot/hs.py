#!/usr/bin/env python3

from pot import Potential
from dataclasses import dataclass, field
from grid import Grid
import numpy as np
from itertools import product

@dataclass
class HardSpheres(Potential):
    grid: Grid
    params: list = field(init=False, default_factory=list)
    func: np.ndarray = field(init=False)

    def _calculate(self, sigma):
        return np.where((self.grid.r_grid >= sigma), 0, 1E30)

    def _mixing(self):
        self.params = list(map(self._arithmetic_mean, product(self.params, repeat=2)))

    def _arithmetic_mean(self, param_tuple):
        sig1, sig2 = param_tuple

        return (0.5 * (sig1 + sig2), )

    def add_param(self, param_tuple):
        self.params.append(param_tuple)
        return self

    def tabulate(self):
        self._mixing()
        funcs = list(map(lambda param: self._calculate(param[0]), self.params))
        self.func = np.swapaxes(np.array(funcs), 1, 0)
        return self

if __name__ == "__main__":
    new_grid = Grid(16384, 20.48)
    HS = HardSpheres(new_grid) \
        .add_param((0.8))      \
        .add_param((1.2))      \
        .tabulate()
    print(HS.func)
    print(HS.func.shape)

    import matplotlib.pyplot as plt

    plt.xlim(0, 2)
    plt.ylim(-1, 5)
    plt.plot(new_grid.r_grid, HS.func[..., 0])
    plt.show()
