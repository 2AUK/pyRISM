import numpy as np
from dataclasses import dataclass, field


@dataclass
class Species(object):

    species_name: str
    atom_sites: list = field(default_factory=list)

    def add_site(self, atom_site):
        self.atom_sites.append(atom_site)
