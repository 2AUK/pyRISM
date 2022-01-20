import numpy as np
from .Ng import NgSolver
from .MDIIS import MDIIS


class Solver(object):
    solver_dispatcher = {
        "Ng": NgSolver,
        "MDIIS": MDIIS,
                         }

    def __init__(self, solv):
        self.solver = solv

    def get_solver(self):
        return self.solver_dispatcher[self.solver]
