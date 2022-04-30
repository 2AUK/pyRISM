#!/usr/bin/env python3
from dataclasses import dataclass, field
from enum import Enum
from mol import InteractionSiteMolecule, SpeciesKind
import toml
from .pairtable import PairTable

class ProblemKind(Enum):
    vv = 0
    uv = 1

@dataclass
class RISMProblem:
    """
    A fully defined RISM integral equation problem.
    On initialisation, it will initialise any and all problem inputs

    Attributes
    ----------
    species: list
    A list containing all InteractionSiteMolecules used in the problem
    problemtype: ProblemKind
    Defines whether the problem is solvent-solvent or solute-solvent.
    This is checked by <handler method for following ISMolecule tags>
    """
    species: list = field(default_factory=list)
    pairs: PairTable = field(default=None)
    problemtype: ProblemKind = field(default=None)

    @classmethod
    def from_toml(self, input_toml):
        inp = toml.load(input_toml)
        problemtype = ProblemKind.vv
        solv_dict_list = inp["solvent"]
        solv_list = self._parse_species_dict(self, solv_dict_list, SpeciesKind.Solvent)
        if "solvent" in inp:
            problemtype = ProblemKind.uv
            solu_dict_list = inp["solute"]
            solu_list = self._parse_species_dict(self, solu_dict_list, SpeciesKind.Solute)
        species_list = [*solv_list, *solu_list]
        return RISMProblem(
            species_list,
            PairTable.generate_tables(species_list),
            problemtype
        )

    def _parse_species_dict(self, input_dict, moltype):
        return list(
            map(
                lambda item: InteractionSiteMolecule.from_dict(
                    item,
                    moltype
                ),
                input_dict
            )
        )

    def add_species(self, species):
        self.species.append(species)
        return self

    def initialise(self):
        """
        Perform checks, generate PairTable, initialise functions.
        """
        pass

if __name__ == "__main__":
    new_RISM = RISMProblem.from_toml("data/Me_Eth_water.toml")
    print(new_RISM.pairs)
