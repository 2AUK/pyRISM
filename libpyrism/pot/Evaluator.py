#!/usr/bin/env python3

from grid import Grid
from dataclasses import dataclass, field
import numpy as np
from pot import Potential

@dataclass
class SystemPotential():
    tot: np.ndarray = field(init=False)
    sh: np.ndarray = field(init=False)
    lo: np.ndarray = field(init=False)
    qtot: np.ndarray = field(init=False)
    qsh: np.ndarray = field(init=False)
    qlo: np.ndarray = field(init=False)

@dataclass
class PotentialEvaluator():
    short_potential: Potential = field(init=False)
    long_potential: Potential = field(init=False)
    q_potential: Potential = field(init=False)
    q_long_potential: Potential = field(init=False)

    def add_short_pot(self, potential):
        self.short_potential = potential
        return self

    def add_long_pot(self, potential):
        self.long_potential = potential
        return self
