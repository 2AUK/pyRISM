#!/usr/bin/env python3

from Core import RISM_Obj
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ClosureObject:

    data: RISM_Obj

    def compute_GF(self, p, energy_unit):
        pass

    def __GF_impl(self, t, h, c, B, T, r, dr, p, energy_unit):
        mu = 4.0 * np.pi * p * dr * np.sum(np.power(r, 2)[:, None, None] * ((0.5 * c * h) - c))
        return mu / B * energy_unit
