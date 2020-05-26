#!/usr/bin/env python3
"""
molecule.py
Stores molecular information as well matrices pertaining to a molecule
"""

from atom import Atom

class Molecule:

    def __init__(self, atoms=[], solv=True):
        self.atoms = atoms
        self.solv = solv

    def append_atom(self, Atom):
        self.atoms.append(Atom)
