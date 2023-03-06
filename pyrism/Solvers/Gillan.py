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
    costab: np.ndarray = field(init=False)

    def kron_delta(self, i, j):
        if i == j:
            return 1.0
        else:
            return 0.0

    def tabulate_cos(self):
        npts = self.data_vv.grid.npts
        ns1 = self.data_vv.ns1
        ns2 = self.data_vv.ns2
        costab = np.zeros((npts, ns1, ns2, ns1, ns2), dtype=np.float64)
        dk = self.data_vv.grid.d_k
        w = self.data_vv.w
        c = self.data_vv.c
        M1 = np.zeros_like(self.data_vv.w)
        M2 = np.zeros_like(self.data_vv.w)

        I = np.identity(ns1, dtype=np.float64)

        coskr = np.cos(self.data_vv.grid.ri * self.data_vv.grid.ki)

        for l in range(npts):
            M1[l] = np.linalg.inv((I - w[l] @ c[l])) @ w[l]
            M2[l] = np.linalg.inv((I - w[l] @ c[l])) @ w[l]

        for i, j in np.ndindex(ns1, ns2):
            for k, l in np.ndindex(ns1, ns2):
                kron1 = self.kron_delta(i, j)
                kron2 = self.kron_delta(k, l)
                costab[:, i, j, k, l] = coskr * (M1[:, i, j] * M2[:, k, l] - kron1 * kron2)

        costab = costab.sum(axis=0) * dk

        self.costab = costab

    def solve(self, RISM, Closure, lam, verbose=False):
        npts = self.data_vv.grid.npts
        ns1 = self.data_vv.ns1
        ns2 = self.data_vv.ns2
        r = self.data_vv.grid.ri
        t = self.data_vv.t
        node = int(npts / self.nbasis / 2)
        node_index = np.arange(0, self.nbasis+1) * node
        nodes = r[node_index]


        P = np.zeros((npts, self.nbasis+1), dtype=np.float64)
        Q = np.zeros_like(P)
        R = np.zeros((self.nbasis, self.nbasis), dtype=np.float64)
        B = np.zeros_like(R)
        A = np.zeros((ns1, ns2, self.nbasis), dtype=np.float64)

        P[..., 0] = 1
        for idx in range(0, npts):
            if nodes[0] <= r[idx] <= nodes[1]:
                P[idx, 1] = (nodes[1] - r[idx]) / nodes[1]

        for a in range(2, self.nbasis+1):
            for idx in range(0, npts):
                if nodes[a-2] <= r[idx] <= nodes[a-1]:
                    P[idx, a] = (r[idx] - nodes[a-2]) / (nodes[a-1] - nodes[a-2])
                elif nodes[a-1] <= r[idx] <= nodes[a]:
                    P[idx, a] = (nodes[a] - r[idx]) / (nodes[a] - nodes[a-1])

        # for a in range(1, self.nbasis+1):
        #       plt.plot(r, P[..., a])
        # plt.show()

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

        identity = np.identity(self.nbasis, dtype=np.float64)
        assert_allclose(kron, identity, atol=1e-4, rtol=1e-4)

        for a in range(self.nbasis):
            for m, n in np.ndindex(ns1, ns2):
                A[m, n, a] = (Q[..., a] * t[:, m, n]).sum()

        while i < self.max_iter:

            # N-R loop
            while True:

                break




    def solve_uv(self, RISM, Closure, lam):
        pass
