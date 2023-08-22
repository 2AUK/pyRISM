#!/usr/bin/env python

from pyrism.Core import RISM_Obj
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import newton_krylov
from .Solver_object import *
import matplotlib.pyplot as plt
from numba import njit, prange

@dataclass
class Gillan(SolverObject):
    nbasis: int = field(default=4)
    costab: np.ndarray = field(init=False)

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def kron_delta(self, i, j):
        if i == j:
            return 1.0
        else:
            return 0.0

    def D(self, i):
        npts = self.data_vv.grid.npts
        ns1 = self.data_vv.ns1
        ns2 = self.data_vv.ns2
        dk = self.data_vv.grid.d_k
        w = self.data_vv.w
        ck = np.zeros_like(self.data_vv.c)
        r_grid = self.data_vv.grid.ri
        k_grid = self.data_vv.grid.ki

        for i, j in np.ndindex(ns1, ns2):
            ck[..., i, j] = self.data_vv.grid.dht(self.data_vv.c[..., i, j])

        # for l in range(npts):
        #     M1[l] = np.linalg.inv((I - w[l] @ ck[l] @ self.data_vv.p)) @ w[l]
        #     M2[l] = np.linalg.inv((I - w[l] @ ck[l] @ self.data_vv.p)) @ w[l]
        #     coskr += np.cos(k_grid[l] * r_grid[i])

        # for i, j in np.ndindex(ns1, ns2):
        #     for k, l in np.ndindex(ns1, ns2):
        #         kron1 = self.kron_delta(i, j)
        #         kron2 = self.kron_delta(k, l)
        #         D[:, i, j, k, l] = coskr[:] * (M1[:, i, j] * M2[:, k, l] - kron1 * kron2)

        D = D_calc(ns1, ns2, npts, w, ck, self.data_vv.p, k_grid, r_grid, i)

        return D.sum(axis=0) * dk

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

        dydc = np.zeros((npts, ns1, ns2, ns1, ns2), dtype=np.float64)

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

        # Checking for positive-definiteness
        #print(self.check_symmetric(R))
        #print(np.linalg.eigvalsh(R))

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

@njit
def D_calc(ns1, ns2, npts, w, ck, p, k_grid, r_grid, dk, i):

    D = np.zeros((npts, ns1, ns2, ns1, ns2), dtype=np.float64)
    coskr = np.zeros((npts), dtype=np.float64)
    M1 = np.zeros_like(w)
    M2 = np.zeros_like(w)
    I = np.identity(ns1, dtype=np.float64)

    for l in prange(npts):
        M1[l] = np.linalg.inv((I - w[l] @ ck[l] @ p)) @ w[l]
        M2[l] = np.linalg.inv((I - w[l] @ ck[l] @ p)) @ w[l]
        coskr[l] = np.cos(k_grid[l] * r_grid[i])

    kron1, kron2 = 0.0, 0.0
    for i in prange(ns1):
        for j in prange(ns2):
            if i == j:
                kron1 = 1.0
            else:
                kron1 = 0.0
            for k in prange(ns1):
                for l in prange(ns2):
                    if k == l:
                        kron2 = 1.0
                    else:
                        kron2 = 0.0
                    #print(i, j, k, l)
                    D[:, i, j, k, l] = coskr[:] * (M1[:, i, j] * M2[:, k, l] - kron1 * kron2)

    return D.sum(axis=0) * dk

