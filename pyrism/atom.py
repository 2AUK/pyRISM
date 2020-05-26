#!/usr/bin/env python3
"""
atom.py
Stores atomic information in a handy object
"""

class Atom:

    def __init__(self, ind, resid, eps, sig, charge):
        self.ind = ind
        self.resid = resid
        self.eps = eps
        self.sig = sig
        self.charge = charge
