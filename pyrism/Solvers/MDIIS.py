#!/usr/bin/env python3

from pyrism.Core import RISM_Obj
import numpy as np
from dataclasses import dataclass, field
from .Solver_object import *
from numba import njit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import sys
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

np.set_printoptions(edgeitems=30, linewidth=180,
    formatter=dict(float=lambda x: "%.5g" % x))

@dataclass
class MDIIS(SolverObject):
    m: int = field(default=1) # depth
    mdiis_damping: float = field(default=0.5)
    fr: list = field(init=False, default_factory=list)
    res: list = field(init=False, default_factory=list)
    RMS_res: list = field(init=False, default_factory=list)

    def step_Picard(self, curr, prev):
        self.fr.append(curr.flatten())
        self.res.append((curr - prev).flatten())
        return prev + self.damp_picard * (curr - prev)

    def step_MDIIS(self, curr, prev, gr):
        return step_MDIIS_impl(curr,
                               prev,
                               self.m,
                               self.res,
                               self.fr,
                               self.damp_picard,
                               self.mdiis_damping,
                               gr)

    def solve(self, RISM, Closure, lam, verbose=False):
        i: int = 0
        A = np.zeros((2, 2), dtype=np.float64)
        b = np.zeros(2, dtype=np.float64)

        if verbose == True:
            print("\nSolving solvent-solvent RISM equation...\n")
        self.fr.clear()
        self.res.clear()
        self.RMS_res.clear()
        while i < self.max_iter:
            c_prev = self.data_vv.c
            try:
                RISM()
                c_A = Closure(self.data_vv)
            except FloatingPointError as e:
                print("Possible divergence")
                print("iteration: {i}".format(i=i))
                print("diff: {diff}".format(diff=(c_A-c_prev).sum()))
                raise e
            if len(self.fr) < self.m:
                c_next = self.step_Picard(c_A, c_prev)
                RMS = np.sqrt(
                1 / self.data_vv.ns1 / self.data_vv.grid.npts * np.power((c_A-c_prev).sum(), 2)
                )
                self.RMS_res.append(RMS)
            else:
                c_next = self.step_MDIIS(c_A, c_prev, self.data_vv.t + c_A)
                c_next = np.reshape(c_next, c_prev.shape)
                RMS = np.sqrt(
                1 / self.data_vv.ns1 / self.data_vv.grid.npts * np.power((c_A-c_prev).sum(), 2)
                )
                if RMS > 10 * min(self.RMS_res):
                    min_index = self.RMS_res.index(min(self.RMS_res))
                    c_next = np.reshape(self.fr[min_index], c_prev.shape)
                    self.fr.clear()
                    self.res.clear()
                    self.RMS_res.clear()
                self.RMS_res.append(RMS)
                self.RMS_res.pop(0)


            self.data_vv.c = c_next
            if self.converged(c_next, c_prev) and verbose == True:
                self.epilogue(i, lam)
                break
            elif self.converged(c_next, c_prev):
                break

            i += 1

            if i == self.max_iter and verbose == True:
                self.epilogue(i, lam)
                raise RuntimeError("max iteration reached")
            elif i == self.max_iter:
                raise RuntimeError("max iteration reached")

    def solve_uv(self, RISM, Closure, lam, verbose=False):
        i: int = 0
        A = np.zeros((2, 2), dtype=np.float64)
        b = np.zeros(2, dtype=np.float64)

        if verbose == True:
            print("\nSolving solute-solvent RISM equation...\n")
        self.fr.clear()
        self.res.clear()
        self.RMS_res.clear()

        while i < self.max_iter:
            c_prev = self.data_uv.c.copy()
            try:
                RISM()
                c_A = Closure(self.data_uv)
            except FloatingPointError as e:
                print("Possible divergence")
                print("iteration: {i}".format(i=i))
                print("diff: {diff}".format(diff=(c_A-c_prev).sum()))
                raise e
            if len(self.fr) < self.m:
                c_next = self.step_Picard(c_A, c_prev)
                RMS = np.sqrt(
                1 / self.data_vv.ns1 / self.data_vv.grid.npts * np.power((c_A-c_prev).sum(), 2)
                )
                self.RMS_res.append(RMS)
            else:
                c_next = self.step_MDIIS(c_A, c_prev, self.data_uv.t + c_A)
                c_next = np.reshape(c_next, c_prev.shape)
                RMS = np.sqrt(
                1 / self.data_uv.ns1 / self.data_uv.grid.npts * np.power((c_A-c_prev).sum(), 2)
                )
                if RMS > 10 * min(self.RMS_res):
                    min_index = self.RMS_res.index(min(self.RMS_res))
                    c_next = np.reshape(self.fr[min_index], c_prev.shape)
                    self.fr.clear()
                    self.res.clear()
                    self.RMS_res.clear()
                self.RMS_res.append(RMS)
                self.RMS_res.pop(0)
            self.data_uv.c = c_next

            if self.converged(c_next, c_prev) and verbose == True:
                self.epilogue(i, lam)
                break
            elif self.converged(c_next, c_prev):
                break

            i += 1

            if i == self.max_iter and verbose == True:
                self.epilogue(i, lam)
                raise RuntimeError("max iteration reached")
            elif i == self.max_iter:
                raise RuntimeError("max iteration reached")

@njit
def step_MDIIS_impl(curr, prev, m, res, fr, damp_picard, damp_mdiis, gr):
    A = np.zeros((m+1, m+1), dtype=np.float64)
    b = np.zeros(m+1, dtype=np.float64)

    b[m] = -1

    for i in prange(m+1):
        A[i, m] = -1
        A[m, i] = -1

    A[m, m] = 0

    for i, j in np.ndindex((m, m)):
        A[i,j] = res[i] @ res[j]

    coef = np.linalg.solve(A, b)

    c_A = np.zeros_like(fr[0])
    min_res = np.zeros_like(fr[0])
    denom = np.sqrt(1.0 + np.power(gr.flatten(), 2))
    for i in prange(m):
        c_A += coef[i] * fr[i]
        min_res += (coef[i] * res[i]) / denom

    c_new = c_A + damp_mdiis * min_res

    fr.append(curr.flatten())
    res.append((curr - prev).flatten())

    fr.pop(0)
    res.pop(0)

    return c_new
