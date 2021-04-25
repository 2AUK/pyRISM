import numpy as np
from utils import *

class Potential_Short:

    potential_table = {
        "LJ"    : [Lennard_Jones, Lorentz_Berthelot]
        "LJ_AB" : [Lennard_Jones_AB, AB_mixing]
        "HS"    : [hard_spheres, arithmetic_mean]
    }

    def __init__(self, rism_obj):
        