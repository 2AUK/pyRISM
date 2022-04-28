#!/usr/bin/env python3
from dataclasses import dataclass, field
from enum import Enum
from .site import InteractionSite
import toml

class SpeciesKind(Enum):
    Solvent = 0
    Solute = 1

@dataclass
class InteractionSiteMolecule:
    name: str
    moltype: SpeciesKind
    nsites: int
    density: float
    sites: list = field(init=False, default_factory=list)

    def add_site(self, site):
        self.sites.append(site)
        return self

    def read_from_toml(self, input_toml):
        pass

    def read_from_dict(self, input_dict):
        pass

    def check(self):
        if len(self.sites) != self.nsites:
            raise ValueError("Mismatch between nsites and length of sites list!")
        return self

    def __lt__(self, other):
        return self.SpeciesKind < other.SpeciesKind


if __name__ == "__main__":
    modified_spce = InteractionSiteMolecule("Modified SPC/E",
                                            SpeciesKind.Solvent,
                                            3,
                                            0.03334) \
                                            .add_site(
                                                InteractionSite("O",
                                                                [0.0, 0.0, 0.0],
                                                                (78.2003154916, 3.166),
                                                                -0.8476)
                                            ) \
                                            .add_site(
                                                InteractionSite("H",
                                                                [1.0, 0.0, 0.0],
                                                                (23.1480985368, 0.8),
                                                                0.4238)
                                            ) \
                                            .add_site(
                                                InteractionSite("H",
                                                                [-0.333314, 0.942816, 0.0],
                                                                (23.1480985368, 0.8),
                                                                0.4238)
                                            ) \
                                            .check()

    print(modified_spce)
