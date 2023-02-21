#!/usr/bin/env python

from Core import RISM_Obj
import numpy as np
from scipy.optimize import newton_krylov
from .Solver_object import *

@dataclass
class Gillan(SolverObject):
    nbasis: int = field(default=4)

    def solve(self, RISM, Closure, lam):
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


        P = np.zeros((npts, ns1, ns2, self.nbasis+1), dtype=np.float64)
        Q = np.zeros_like(P)
        R = np.zeros((self.nbasis, self.nbasis), dtype=np.float64)
        B = np.zeros_like(R)
        A = np.zeros((self.nbasis), dtype=np.float64)

        P[..., 0] = 1
        for idx in range(0, npts):
            if nodes[0] <= t[idx, ...] <= nodes[1]:
                P[idx, 1] = (nodes[1] - t[idx, ...]) / (nodes[1])

        for a in range(2, nbasis+1):
            for idx in range(0, npts):
                if nodes[a-2] <= t[idx] <= nodes[a-1]:
                    P[idx, a] = (t[idx] - nodes[a-2])/(nodes[a-1] - nodes[a-2])
                elif nodes[a-1] <= t[idx] <= nodes[a]:
                    P[idx, a] = (nodes[a] - t[idx])/(nodes[a] - nodes[a-1])

        P_proper = P[..., 1:].copy()
        P_a = P_proper.copy()
        P_b = P_proper.copy()

        for a in range(nbasis):
            for b in range(nbasis):
                R[a,b] = (Pa[..., a] * Pb[..., b]).sum()

        B = np.linalg.inv(R)

        for i in range(npts):
            for a in range(nbasis):
                Q[i,a] = (B[a,:] * P_skip_zero[i,:]).sum()

        for a in range(nbasis):
            A[a] = (Q[..., a] * t).sum()

        coarse = np.zeros_like(t)




    def solve_uv(self, RISM, Closure, lam):
        pass
