#!/usr/bin/env python3

from dataclasses import dataclass, field
from itertools import product, chain

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

    def generate_tables(self):
        self.species.sort()
        all_species = chain(*[x.site for x in self.species])
        solvent_species = chain(*[x.site for x in self.species if x.speciestype == SpeciesKind.solute])
        solute_species = chain(*[x.site for x in self.species if x.speciestype == SpeciesKind.Solute])
        self.upairs = self._pairs_in_list(solvent_species)
        self.vpairs = self._pairs_in_list(solute_species)
        self.uvpairs = self._pairs_in_list(all_species)

    def iter_solute_pairs(self):
        return iter(self.upairs)

    def iter_solvent_pairs(self):
        return iter(self.vpairs)

    def iter_all_pairs(self):
        return iter(self.uvpairs)

    def _pairs_in_list(self, input_list):
        return list(product(self.params, repeat=2))

    def _filter_species(self, input_list, filter):
        for x in input_list:
            if filter(x):
                return true
            return false
