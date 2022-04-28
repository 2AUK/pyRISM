#!/usr/bin/env python3

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class SiteKind(Enum):
    atom_centered = 0
    auxilliary = 1

@dataclass
class InteractionSite:
    name: str
    coordinates: np.ndarray
    parameters: tuple
    charge: float
    sitetype: SiteKind = field(default=SiteKind.atom_centered)

    def __post_init__(self):
        self._check()

    def _check(self):
        if coordinates.size != 3:
            raise ValueError("Require x, y and z coordinates!")
        if 0 in self.parameters:
            raise ValueError("Site {name} has a parameter set to 0!".format(name=self.name))

if __name__ == "__main__":
    new_good_site = InteractionSite("H", [1.0, 0.0, 0.0], (78.15, 0.4), 0.0)
    print(new_good_site)
    new_bad_site = InteractionSite("X", [0, 0, 0], (0.0, 0.8), 0.0, SiteKind.auxilliary)
