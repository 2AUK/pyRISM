#!/usr/bin/env python3

from pot import Potential
from dataclasses import dataclass, field
from grid import Grid
import numpy as np
from itertools import product


class SquareWell(Potential):
    grid: Grid
    params: list = field(init=False, default_factory=list)
    func: np.ndarray = field(init=False)

    def _calculate(self, epsilon, sigma):
        condlist = [ self.grid.r_grid < sigma,
                     (self.grid.r_grid >= sigma) & (self.grid.r_grid < (sigma + 1.5 * sigma)) ]
        choicelist = [ 1E30, -epsilon ]
        return np.select(condlist, choicelist, default=0.0)

    def _mixing(self):
        self.params = list(map(self._arithmetic_mean, product(self.params, repeat=2)))

    def _arithmetic_mean(self, param_tuple):
        p1, p2 = param_tuple
        eps1, sig1 = p1
        eps2, sig2 = p2

        return (0.5 * (eps1 + eps2), 0.5 * (sig1 + sig2))

    def add_param(self, param_tuple):
        self.params.append(param_tuple)
        return self

    def tabulate(self):
        self._mixing()
        funcs = list(map(lambda param: self._calculate(param[0], param[1]), self.params))
        self.func = np.swapaxes(np.array(funcs), 1, 0)
        return self


if __name__ == "__main__":
    new_grid = Grid(16384, 20.48)
    SW = SquareWell(new_grid) \
        .add_param((5, 0.8))  \
        .add_param((12, 1.2))  \
        .tabulate()
    print(SW.func)
    print(SW.func.shape)

    import matplotlib.pyplot as plt

    plt.xlim(0, 5)
    plt.ylim(-10, 10)
    plt.plot(new_grid.r_grid, SW.func[..., 2])
    plt.show()
