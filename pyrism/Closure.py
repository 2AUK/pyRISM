import numpy as np
import Data
from closure_routines import *


@attr.s
class Closure:

    closure_dispatcher = {
        "HNC": HyperNetted_Chain,
        "KH": KovalenkoHirata,
        "PSE": PSE_n,
        "PY": PercusYevick,
    }

    closure: str = attr.ib()

    @property
    def closure(self):
        return closure_dispatcher[self.closure]