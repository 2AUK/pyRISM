#!/usr/bin/env python3

from pot import Potential
from dataclasses import dataclass, field
from grid import Grid
import numpy as np
from itertools import product

@dataclass
class LennardJones(Potential):
    grid: Grid
    params: list[tuple, ...] = field(init=False, default_factory=list)
    func: np.ndarray = field(init=False)

    def _calculate(self, epsilon, sigma):
        return 4.0 * epsilon * \
            ((sigma / self.grid.r_grid) ** 12 - \
             (sigma / self.grid.r_grid) ** 6)

    def _mixing(self):
        self.params = list(map(self._LorentzBerthelot, product(self.params, repeat=2)))

    def _LorentzBerthelot(self, param_tuple):
        p1, p2 = param_tuple
        eps1, sig1 = p1
        eps2, sig2 = p2

        return (np.sqrt(eps1 * eps2), 0.5 * (sig1 + sig2))

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
    LJ = LennardJones(new_grid)  \
        .add_param((78.15, 0.8)) \
        .add_param((23.15, 0.4)) \
        .add_param((23.15, 0.4)) \
        .tabulate()
    print(LJ.func)
    print(LJ.func.shape)

    import matplotlib.pyplot as plt

    plt.xlim(0, 5)
    plt.ylim(-90, 20)
    plt.plot(new_grid.r_grid, LJ.func[..., 0])
    plt.show()
