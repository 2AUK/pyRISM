import numpy as np
from utils import *

class Potential_Long:

    potential_table = {
        "cou"   : coulomb
        "erfr"  : coulomb_lr_r
        "erfk"  : coulomb_lr_k
    }

    def __init__(self, rism_obj, pot):
        self.rism = rism_obj
        self.pot = pot

    def solve(self):
        return potential_table[self.pot](self.rism)