#!/usr/bin/env python3

from Core import RISM_Obj
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ClosureObject:

    dat_vv: RISM_Obj
    dat_uv: RISM_Obj = None

    def compute_GF(self, p, energy_unit):
        if self.dat_uv is not None:
            return self.__GF_impl(self.dat_uv.t, self.dat_uv.h, self.dat_uv.c,
                                  self.dat_uv.B, self.dat_uv.T, self.dat_uv.grid.ri,
                                  self.dat_uv.grid.d_r, p, energy_unit)
        else:
            raise RuntimeError("No free energy functional for solvent-solvent interactions implemented.")

    def __GF_impl(self, t, h, c, B, T, r, dr, p, energy_unit):
        mu = 4.0 * np.pi * p * dr * np.sum(np.power(r, 2)[:, None, None] * ((0.5 * c * h) - c))
        return mu / B * energy_unit
