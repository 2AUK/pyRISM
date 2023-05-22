import numpy as np
from dataclasses import dataclass, field


@dataclass
class Species(object):

    species_name: str
    dens: float = field(init=False)
    ns: int = field(init=False)
    atom_sites: list = field(default_factory=list)

    def add_site(self, atom_site):
        self.atom_sites.append(atom_site)

    def set_density(self, density):
        self.dens = density

    def set_numsites(self, ns):
        self.ns = ns
