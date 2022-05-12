#!/usr/bin/env python3
from dataclasses import dataclass, field
from enum import Enum
from mol import InteractionSiteMolecule, SpeciesKind
import toml
from .pairtable import PairTable
from grid import Grid
import numpy as np
from .solution import RISMSolution
from .functions import RISMFunctions

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
    grid: Grid
    kT: float
    _temp: float
    _beta: float = field(init=False)
    species: list = field(default_factory=list)
    pairs: PairTable = field(default=None)
    problemtype: ProblemKind = field(default=None)
    subproblem: RISMSolution = field(init=False, default=None)

    def __post_init__(self):
        self._beta = 1 / self.kT / self._temp

    @classmethod
    def from_toml(self, input_toml):
        inp = toml.load(input_toml)
        grid = Grid(inp["system"]["npts"], inp["system"]["radius"])
        problemtype = ProblemKind.vv
        solv_dict_list = inp["solvent"]
        solv_list = self._parse_species_dict(self, solv_dict_list, SpeciesKind.Solvent)
        if "solvent" in inp:
            problemtype = ProblemKind.uv
            solu_dict_list = inp["solute"]
            solu_list = self._parse_species_dict(self, solu_dict_list, SpeciesKind.Solute)
        species_list = [*solv_list, *solu_list]

        return RISMProblem(
            grid,
            inp["system"]["kT"],
            inp["system"]["temp"],
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

    @property
    def temp(self):
        return self._temp

    @property
    def beta(self):
        return self._temp

    @temp.setter
    def temp(self, val):
        self._temp = val
        self._beta = 1 / kT / self._temp

    def initialise(self):
        """
        Perform checks, generate PairTable, initialise functions.
        """
        pass


if __name__ == "__main__":
    new_RISM = RISMProblem.from_toml("data/Me_Eth_water.toml")
    print(new_RISM.pairs)
    print(new_RISM.grid)
