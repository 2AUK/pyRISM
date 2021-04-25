import numpy as np
from closure_routines import *


class Closure:

    closure_table = {
        "HNC": HyperNetted_Chain,
        "KH": KovalenkoHirata,
        "PSE": PSE_n,
        "PY": PercusYevick,
    }

    def __init__(self, rism_obj):
        self.rism = rism_obj

    def solve(self):
        return closure_table[self.rism.closure](self.rism)