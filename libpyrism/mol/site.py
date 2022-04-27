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
    sitetype: SiteKind

    def _parameter_check(self):
        if 0 in self.parameters:
            raise ValueError("Site {name} has a parameter set to 0!".format(name=self.name))

    def __post_init__(self):
        self._parameter_check()


if __name__ == "__main__":
    new_good_site = InteractionSite("H", [1.0, 0.0, 0.0], (78.15, 0.4), SiteKind.atom_centered)
    print(new_good_site)
    new_bad_site = InteractionSite("X", [0, 0, 0], (0.0, 0.8), SiteKind.atom_centered)
