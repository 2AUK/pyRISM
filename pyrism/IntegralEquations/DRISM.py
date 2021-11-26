#!/usr/bin/env python3

import numpy as np
from Core import RISM_Obj
from dataclasses import dataclass, field
import Util

@dataclass
class DRISM(object):

    data_vv: RISM_Obj
    data_uv: RISM_Obj = None
    h_c0: np.ndarray = None

    def compute_vv(self):
        pass

    def compute_uv(self):
        pass

    def calculate_y(self):
        total_density = 0
        dm, _ = Util.dipole_moment(self.data_vv)
        for isp in self.data_vv.species:
            total_density += isp.dens
        dmdensity = total_density * dm * dm

        return 4.0 * np.pi * self.data_vv.B * dmdensity / 9.0


def vv_impl():
    pass

def uv_impl():
    pass
