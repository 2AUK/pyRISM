#!/usr/bin/env python3
from dataclasses import dataclass, field
from enum import Enum

class Species(Enum):
    Solvent = 0
    Solute = 1


@dataclass
class InteractionSiteMolecule:
    speciestype: SpeciesKind
    nsites: int
    sites: list = field(init=False, default_factory=list)

    def add_site(self):
        return self


if __name__ == "__main__":
    pass
