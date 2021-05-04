import numpy as np
import attr
import Data
from closure_routines import *


@attr.s
class Closure:

    closure_dispatcher = {
        "HNC": HyperNetted_Chain,
        "KH": KovalenkoHirata,
        "PSE-1": PSE_1,
        "PSE-2": PSE_2,
        "PSE-3": PSE_3
        "PY": PercusYevick,
    }

    closure: str = attr.ib()

    @property
    def closure(self):
        return closure_dispatcher[self.closure]