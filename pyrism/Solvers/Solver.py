import numpy as np
from .Ng import NgSolver
from .MDIIS import MDIIS
from .Picard import Picard
from .Gillan import Gillan


class Solver(object):
    solver_dispatcher = {
        "Picard": Picard,
        "Ng": NgSolver,
        "MDIIS": MDIIS,
        "Gillan": Gillan,
                         }

    def __init__(self, solv):
        self.solver = solv

    def get_solver(self):
        return self.solver_dispatcher[self.solver]
