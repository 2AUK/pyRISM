import numpy as np
from .potential_routines import *
from .mixing import *


class Potential:

    potential_dispatcher = {
        "LJ": [Lennard_Jones, Lorentz_Berthelot],
        "LJ_AB": [Lennard_Jones_AB, AB_mixing],
        "HS": [hard_spheres, arithmetic_mean],
        "cou": [coulomb, None],
        "erfr": [coulomb_lr_r, None],
        "erfk": [coulomb_lr_k, None],
    }

    def __init__(self, pot):
        self.potential = pot

    def get_potential(self):
        return self.potential_dispatcher[self.potential]
