#!/usr/bin/env python3

import numpy as np
from Core import RISM_Obj
from dataclasses import dataclass, field
import Util

@dataclass
class DRISM(object):

    data_vv: RISM_Obj
    data_uv: RISM_Obj = None
    h_k: np.ndarray = None
    h_c0: float = None
    y: float = None
    diel: float = None

    def compute_vv(self):
        pass

    def compute_uv(self):
        pass

    def calculate_DRISM_params(self):
        total_density = 0
        dm, _ = Util.dipole_moment(self.data_vv)
        for isp in self.data_vv.species:
            total_density += isp.dens
        dmdensity = total_density * dm * dm
        self.y = 4.0 * np.pi * self.data_vv.B * dmdensity / 9.0
        self.h_c0 = ((self.diel - 1.0) / self.y) * total_density

    def D_matrix(self):
        for k in self.data_vv.grid.ki:
            for isp in self.data_vv.species:
                for iat in isp:
                    k_coord = iat.coords * k
                    if k_coord[0] == 0:
                        pass

    def __post_init__(self):
        self.calculate_DRISM_params()


def vv_impl():
    pass

def uv_impl():
    pass
