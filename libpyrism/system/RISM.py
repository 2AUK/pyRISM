#!/usr/bin/env python3
from dataclasses import dataclass, field
from enum import Enum

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
    species: list = field(init=False, default_factory=list)
    pairs: PairTable = field(init=False)
    problemtype: ProblemKind = field(init=False)

    def add_species(self, species):
        self.species.append(species)
        return self

    def initialise(self):
        """
        Perform checks, generate PairTable, initialise functions.
        """
        pass
