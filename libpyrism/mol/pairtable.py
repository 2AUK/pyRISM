#!/usr/bin/env python3

from dataclasses import dataclass, field
from itertools import product

@dataclass
class PairTable:
    """
    Data structure to keep track of pairs of `InteractionSiteMolecules`.
    This class should only ever by instantiated by a RISMProblem, as it
    will keep track of pairs of `InteractionSites` and which
    `InteractionSiteMolecules` they reside in.

    Attributes
    ----------
    species: list
    A table containing
    """
    species: list
    upairs: list = field(init=False, default_factory=list)
    vpairs: list = field(init=False, default_factory=list)
    uvpairs: list = field(init=False, default_factory=list)

    def generate_table(self):
        self.species.sort()
        solvent_species = [x for x in self.species if x.speciestype == SpeciesKind.Solvent]
        solute_species = [x for x in self.species if x.speciestype == SpeciesKind.Solute]
        self.upairs = self._pairs_in_list(solvent_species)
        self.vpairs = self._pairs_in_list(solute_species)
        self.uvpairs = self._pairs_in_list(self.species)

    def _pairs_in_list(self, input_list):
        return list(product(self.params, repeat=2))
