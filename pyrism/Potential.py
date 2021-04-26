import numpy as np
import Data
from potential_routines import *
from utils import *

@attr.s
class Potential:

    potential_dispatcher = {
        "LJ"    : [Lennard_Jones, Lorentz_Berthelot]
        "LJ_AB" : [Lennard_Jones_AB, AB_mixing]
        "HS"    : [hard_spheres, arithmetic_mean]
        "cou"   : [coulomb, None]
        "erfr"  : [coulomb_lr_r, None]
        "erfk"  : [coulomb_lr_k, None]
    }

    potential: str = attr.ib()

    @property
    def potential(self):
        return potential_dispatcher[self.potential]