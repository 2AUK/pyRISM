#!/usr/bin/env python3

from dataclasses import dataclass, field
from itertools import product, chain
from mol import SpeciesKind

@dataclass
class PairTable:
    """
    Data structure to keep track of pairs of `InteractionSiteMolecules`.
    This class should only ever by instantiated by a RISMProblem, as it
    will keep track of pairs of `InteractionSites` and which
    `InteractionSiteMolecules` they reside in.
    Keep track of pairs of indices only, otherwise Object seems excessively
    large.

    Attributes
    ----------
    species: list
    A table containing
    """
    species: list
    upairs: list = field(init=False, default_factory=list)
    vpairs: list = field(init=False, default_factory=list)
    uvpairs: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.generate_tables()

    def generate_tables(self):
        self.species.sort()
        all_species = chain(*[x.sites for x in self.species])
        solvent_species = chain(*[x.sites for x in self.species if x.moltype == SpeciesKind.Solvent])
        solute_species = chain(*[x.sites for x in self.species if x.moltype == SpeciesKind.Solute])
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
        return list(product(input_list, repeat=2))

    def _filter_species(self, input_list, filter):
        for x in input_list:
            if filter(x):
                return true
            return false
