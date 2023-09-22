import numpy as np
from .closure_routines import *


class Closure(object):
    closure_dispatcher = {
        "HNC": HyperNetted_Chain,
        "KH": KovalenkoHirata,
        "PSE-1": PSE_1,
        "PSE-2": PSE_2,
        "PSE-3": PSE_3,
        "PY": PercusYevick,
        "rHNC": renormalized_HyperNetted_Chain,
        "rPY": renormalized_PercusYevick,
        "rKH": renormalized_KovalenkoHirata,
    }

    def __init__(self, clos):
        self.closure = clos

    def get_closure(self):
        return self.closure_dispatcher[self.closure]
