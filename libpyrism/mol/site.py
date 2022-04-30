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
    parameters: tuple
    coordinates: np.ndarray
    charge: float
    sitetype: SiteKind = field(default=SiteKind.atom_centered)

    def __post_init__(self):
        self.coordinates = np.asarray(self.coordinates)
        self._check()

    @classmethod
    def from_dict(self, input_dict):
        (site_name, site_info), = input_dict.items()
        charge = site_info[0].pop()
        return InteractionSite(site_name,
                               site_info[0],
                               np.asarray(site_info[1]),
                               charge)

    def _check(self):
        if self.coordinates.size != 3:
            raise ValueError("Require x, y and z coordinates!")
        if 0 in self.parameters:
            raise ValueError("Site {name} has a parameter set to 0!".format(name=self.name))

if __name__ == "__main__":
    new_good_site = InteractionSite("H",  (78.15, 0.4), [1.0, 0.0, 0.0], 0.0)
    print(new_good_site)
    new_site_from_dict = InteractionSite.from_dict(
        {'C': [[25.16098, 3.8, 0.15], [0.7031, 0.0083, -0.1305]]})
    print(new_site_from_dict)
