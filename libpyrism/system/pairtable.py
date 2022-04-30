#!/usr/bin/env python3

from dataclasses import dataclass, field
from itertools import product, chain
from mol import SpeciesKind
from copy import deepcopy

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
    upairs: list = field(default_factory=list)
    vpairs: list = field(default_factory=list)
    uvpairs: list = field(default_factory=list)
    upairs_flat: list = field(default_factory=list)
    vpairs_flat: list = field(default_factory=list)

    @classmethod
    def generate_tables(self, species_list):
        species_list.sort()
        all_species = chain(*[[x.sites] for x in species_list])
        solvent_species = chain(*[[x.sites] for x in species_list if x.moltype == SpeciesKind.Solvent])
        solute_species = chain(*[[x.sites] for x in species_list if x.moltype == SpeciesKind.Solute])
        solv_flat = chain(*[x.sites for x in species_list if x.moltype == SpeciesKind.Solvent])
        solu_flat =  chain(*[x.sites for x in species_list if x.moltype == SpeciesKind.Solute])
        upairs = self._pairs_in_list(self, list(solute_species))
        vpairs = self._pairs_in_list(self, list(solvent_species))
        upairs_flat = self._pairs_in_list_flat(self, list(solu_flat))
        vpairs_flat = self._pairs_in_list_flat(self, list(solv_flat))
        uvpairs = self._pairs_in_list_flat(self, list(chain(*(all_species))))

        return PairTable (
            upairs,
            vpairs,
            uvpairs,
            upairs_flat,
            vpairs_flat,
        )


    def _pairs_in_list(self, input_list):
        return [list(product(range(len(item)), repeat=2)) for item in input_list]

    def _pairs_in_list_flat(self, input_list):
        return list(product(range(len(input_list)), repeat=2))

    def _filter_species(self, input_list, filter):
        for x in input_list:
            if filter(x):
                return true
            return false
