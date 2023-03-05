#!/usr/bin/env python

from pyrism.Core import RISM_Obj
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import newton_krylov
from .Solver_object import *
import matplotlib.pyplot as plt

@dataclass
class Gillan(SolverObject):
    nbasis: int = field(default=4)

    def solve(self, RISM, Closure, lam, verbose=False):
        npts = self.data_vv.grid.npts
        ns1 = self.data_vv.ns1
        ns2 = self.data_vv.ns2
        r = self.data_vv.grid.ri
        t = self.data_vv.t
        node = int(npts / self.nbasis / 2)
        node_index = np.arange(0, self.nbasis+1) * node
        nodes = t[node_index, ...]
        fine = t.copy()
        fine[:node_index[-1], ...] = 0

        print(fine)
        print(nodes.shape)


        P = np.zeros((npts, ns1, ns2, self.nbasis+1), dtype=np.float64)
        Q = np.zeros_like(P)
        R = np.zeros((self.nbasis, self.nbasis), dtype=np.float64)
        B = np.zeros_like(R)
        A = np.zeros((self.nbasis), dtype=np.float64)

        P[..., 0] = 1
        for i, m, n in np.ndindex(npts, ns1, ns2):
            if nodes[0, m, n] <= t[i, m, n] <= nodes[1, m, n]:
                P[i, m, n, 1] = (nodes[1, m, n] - t[i, m, n]) / (nodes[1, m, n])

        for a in range(2, self.nbasis+1):
            for i, m, n in np.ndindex(npts, ns1, ns2):
                if nodes[a-2, m, n] <= t[i, m, n] <= nodes[a, m, n]:
                    P[i, m, n, a] = (t[i, m, n] - nodes[a-2, m, n])/(nodes[a-1, m, n] - nodes[a-2, m, n])
                elif nodes[a-1] <= t[i, m, n] <= nodes[a, m, n]:
                    P[i, m, n, a] = (nodes[a, m, n] - t[i, m, n])/(nodes[a, m, n] - nodes[a-1, m, n])

        for a in range(0, self.nbasis+1):
            for m, n in np.ndindex(ns1, ns2):
                print(P[..., m, n, a])
                plt.plot(r, P[..., m, n, a])
        plt.show()

        P_proper = P[..., 1:].copy()
        P_skip_zero = P_proper.copy()
        Pa = P_proper.copy()
        Pb = P_proper.copy()

        for a in range(self.nbasis):
            for b in range(self.nbasis):
                R[a,b] = (Pa[..., a] * Pb[..., b]).sum()

        B = np.linalg.inv(R)

        for i in range(npts):
            for a in range(self.nbasis):
                Q[i,a] = (B[a,:] * P_skip_zero[i,:]).sum()

        kron = np.zeros_like(R)

        for a in range(self.nbasis):
            for b in range(self.nbasis):
                kron[a,b] = (Q[...,a] * P_skip_zero[...,b]).sum()

        identity = np.identity((self.nbasis, self.nbasis), dtype=np.float64)
        assert_allclose(kron, nbasis, atol=1e-4, rtol=1e-4)

        for a in range(nbasis):
            A[a] = (Q[..., a] * t).sum()

        coarse = np.zeros_like(t)




    def solve_uv(self, RISM, Closure, lam):
        pass
