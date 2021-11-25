#!/usr/bin/env python3

import numpy as np
from Core import RISM_Obj
from dataclasses import dataclass, field

@dataclass
class DRISM(object):

    data_vv: RISM_Obj
    data_uv: RISM_Obj = None
    h_c0: np.ndarray = None

    def compute_vv(self):
        pass

    def compute_uv(self):
        pass

def vv_impl():
    pass

def uv_impl():
    pass
