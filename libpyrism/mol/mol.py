#!/usr/bin/env python3
from dataclasses import dataclass, field
from enum import IntEnum
from .site import InteractionSite
import toml

class SpeciesKind(IntEnum):
    Solvent = 0
    Solute = 1

@dataclass
class InteractionSiteMolecule:
    name: str
    moltype: SpeciesKind
    nsites: int
    density: float
    sites: list = field(default_factory=list)

    def add_site(self, site):
        self.sites.append(site)
        return self

    @classmethod
    def from_dict(self, input_dict, moltype):
        sites, = input_dict["sites"]
        site_list = list(map(lambda item: InteractionSite.from_dict({item[0] : item[1]}), sites.items()))
        mol = InteractionSiteMolecule(
            input_dict["name"],
            moltype,
            input_dict["ns"],
            input_dict["dens"],
            site_list
        )
        mol.check()

        return mol

    def check(self):
        if len(self.sites) != self.nsites:
            raise ValueError("Mismatch between nsites ({nsites}) and length of sites list({len_sites})!".format(nsites=self.nsites, len_sites=len(self.sites)))
        return self

    def __gt__(self, other):
        self.moltype > other.moltype

    def __lt__(self, other):
        return self.moltype < other.moltype

if __name__ == "__main__":
    new_mol = InteractionSiteMolecule.from_dict({'name': 'Modified SPC/E',
                                                 'dens': 0.03334,
                                                 'ns': 3,
                                                 'sites':
                                                 [
                                                     {'O_w': [[76.48937, 3.15, -0.834], [0.0, 0.0, 0.0]],
                                                      'H1_w': [[23.1481, 0.4, 0.417], [0.9572, 0.0, 0.0]],
                                                      'H2_w': [[23.1481, 0.4, 0.417], [-0.239988, 0.926627, 0.0]]}]
                                                 },
                                                SpeciesKind.Solvent
                                                )
    print(new_mol)
