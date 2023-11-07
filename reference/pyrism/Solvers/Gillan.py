#!/usr/bin/env python

from pyrism.Core import RISM_Obj
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import newton_krylov
from .Solver_object import *
import matplotlib.pyplot as plt
from numba import njit, prange
import matplotlib.pyplot as plt


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

        ck = self.data_vv.grid.dht(self.data_vv.c)

        # for l in range(npts):
        #     M1[l] = np.linalg.inv((I - w[l] @ ck[l] @ self.data_vv.p)) @ w[l]
        #     M2[l] = np.linalg.inv((I - w[l] @ ck[l] @ self.data_vv.p)) @ w[l]
        #     coskr += np.cos(k_grid[l] * r_grid[i])

        # for i, j in np.ndindex(ns1, ns2):
        #     for k, l in np.ndindex(ns1, ns2):
        #         kron1 = self.kron_delta(i, j)
        #         kron2 = self.kron_delta(k, l)
        #         D[:, i, j, k, l] = coskr[:] * (M1[:, i, j] * M2[:, k, l] - kron1 * kron2)

        D = D_calc(ns1, ns2, npts, w, ck, self.data_vv.p, k_grid, r_grid, dk, i)

        return D

    def E(self, i):
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

        E = E_calc(ns1, ns2, npts, w, ck, self.data_vv.p, k_grid, r_grid, dk, i)

        return E

    def solve(self, RISM, Closure, lam, verbose=False):
        npts = self.data_vv.grid.npts
        ns1 = self.data_vv.ns1
        ns2 = self.data_vv.ns2
        r = self.data_vv.grid.ri
        t = self.data_vv.t
        node = int(npts / self.nbasis / 2)
        node_index = np.arange(0, self.nbasis + 1) * node
        nodes = r[node_index]
        dr = self.data_vv.grid.d_r
        dk = self.data_vv.grid.d_k
        r_grid = self.data_vv.grid.ri
        k_grid = self.data_vv.grid.ki

        P = np.zeros((npts, self.nbasis + 1), dtype=np.float64)
        Q = np.zeros_like(P)
        R = np.zeros((self.nbasis, self.nbasis), dtype=np.float64)
        B = np.zeros_like(R)
        A_prev = np.zeros((self.nbasis, ns1, ns2), dtype=np.float64)
        A_curr = np.zeros((self.nbasis, ns1, ns2), dtype=np.float64)
        A_new = np.zeros((self.nbasis, ns1, ns2), dtype=np.float64)
        A_initial = np.zeros((self.nbasis, ns1, ns2), dtype=np.float64)
        A = np.empty((self.nbasis, ns1, ns2), dtype=np.float64)

        dydy = np.zeros((npts, npts, ns1, ns2, ns1, ns2), dtype=np.float64)
        jac = np.zeros((self.nbasis, self.nbasis, ns1, ns2, ns1, ns2))

        P[..., 0] = 1
        for idx in range(0, npts):
            if nodes[0] <= r[idx] <= nodes[1]:
                P[idx, 1] = (nodes[1] - r[idx]) / nodes[1]

        for a in range(2, self.nbasis + 1):
            for idx in range(0, npts):
                if nodes[a - 2] <= r[idx] <= nodes[a - 1]:
                    P[idx, a] = (r[idx] - nodes[a - 2]) / (nodes[a - 1] - nodes[a - 2])
                elif nodes[a - 1] <= r[idx] <= nodes[a]:
                    P[idx, a] = (nodes[a] - r[idx]) / (nodes[a] - nodes[a - 1])

        # for a in range(1, self.nbasis+1):
        #       plt.plot(r, P[..., a])
        # plt.show()

        P_proper = P[..., 1:].copy()
        P_skip_zero = P_proper.copy()
        Pa = P_proper.copy()
        Pb = P_proper.copy()

        for a in range(self.nbasis):
            for b in range(self.nbasis):
                R[a, b] = (Pa[..., a] * Pb[..., b]).sum()

        # Checking for positive-definiteness
        # print(self.check_symmetric(R))
        # print(np.linalg.eigvalsh(R))

        B = np.linalg.inv(R)

        for i in range(npts):
            for a in range(self.nbasis):
                Q[i, a] = (B[a, :] * P_skip_zero[i, :]).sum()

        kron = np.zeros_like(R)

        for a in range(self.nbasis):
            for b in range(self.nbasis):
                kron[a, b] = (Q[..., a] * P_skip_zero[..., b]).sum()

        

        identity = np.identity(self.nbasis, dtype=np.float64)
        assert_allclose(kron, identity, atol=1e-4, rtol=1e-4)

        idx = 0
        if verbose == True:
            print("\nSolving solvent-solvent RISM equation...\n")
        
        coarse_t = np.zeros_like(self.data_vv.t)
        coarse_t[:node_index[-1], ...] = self.data_vv.t[:node_index[-1], ...]

         # construct initial set of coefficients a
        for a in range(self.nbasis):
            for m, n in np.ndindex(ns1, ns2):
                A[a, m, n] = (Q[..., a] * self.data_vv.t[:, m, n]).sum()


        while idx < self.max_iter:
            print("Iteration: {idx}".format(idx=idx))

            A_prev = A

            sum_term = 0

            P_swapped = np.swapaxes(P_skip_zero, 0, 1)
            
            # new t(r) from a and delta t(r)
            # for first iteration, this shouldn't make any changes
            for i, j, k in np.ndindex(npts, ns1, ns2):
                self.data_vv.t[i, j, k] = (A[:, j, k] * P_swapped[:, i]).sum(axis=0) + coarse_t[i, j, k]
            plt.plot(r_grid, self.data_vv.t[:, 0, 0])
            plt.savefig("tr_iteration_{i}.png".format(i=idx), format="png")

            self.data_vv.c = Closure(self.data_vv)
            previous_coarse_t = coarse_t
            RISM()

            # construct set of coefficients a`
            for a in range(self.nbasis):
                for m, n in np.ndindex(ns1, ns2):
                    A_curr[a, m, n] = (Q[..., a] * self.data_vv.t[:, m, n]).sum()

            idx += 1
            
            print("Performing Newton-Raphson steps")
            g = 0
            #N-R loop
            while True:
                print("--NR Step {i}".format(i=g))
                
                # derivative for HNC closure
                dydc = np.exp(-self.data_vv.B * self.data_vv.u_sr + self.data_vv.t) - 1.0

                for i, j in np.ndindex(npts, npts):
                    if i == 0:
                        dydy[i, j, ...] = 2 * dr / np.pi  * r_grid[j] * self.E(j) * dydc[j]
                    else:
                        dydy[i, j, ...] = dr / np.pi  * r_grid[i] / r_grid[j] * (self.D(i - j) + self.D(i + j)) * dydc[j]

                dada = np.zeros((self.nbasis, self.nbasis, ns1, ns2, ns1, ns2))

                for u, v, i, j, k, l in np.ndindex(self.nbasis, self.nbasis, ns1, ns2, ns1, ns2):
                    #sum_term = 0 
                    #for m, n in np.ndindex(npts, npts):
                    #    sum_term += Q[m, u] * dydy[m, n, i, j, k, l] * P[n, v]
                    dada[u, v, i, j, k, l] =  (Q[np.newaxis, :, u] * dydy[:, :, i, j, k, l] * P_skip_zero[:, np.newaxis, v]).sum(axis=(0, 1))
                    #dada[u, v, i, j, k, l] = sum_term
                
                for u, v, i, j, k, l in np.ndindex(self.nbasis, self.nbasis, ns1, ns2, ns1, ns2):
                    kron1 = self.kron_delta(u, v)
                    kron2 = self.kron_delta(i, j)
                    kron3 = self.kron_delta(k, l)
                    jac[u, v, i, j, k, l] = kron1 * kron2 * kron3 - dada[u, v, i, j, k, l]

                inv_jac = np.zeros((self.nbasis, ns1, ns2))

                for v, k, l in np.ndindex(self.nbasis, ns1, ns2):
                    inv_jac[v, k, l] = jac[:, v, :, :, k, l].sum()

                inv_jac = np.linalg.inv(inv_jac)

                nr_term = np.zeros((self.nbasis, ns1, ns2))

                for u, i, j in np.ndindex((self.nbasis, ns1, ns2)):
                    nr_term[u, i, j] = inv_jac[u, i, j] * (A_curr - A_prev)[u, i, j]

                A_new = A_curr - nr_term

                # A_new = self.step_NR(Q, P_skip_zero, A_curr, A_prev)

                print("----Diff: {diff}".format(diff=np.absolute((A_prev - A_new)).max()))
                if np.absolute((A_prev - A_new)).max() < 1e-5:
                    A = A_new
                    break
                else:
                    A_prev = A_curr
                    A_curr = A_new
                g+=1

            print("\nRunning elementary cycle for next coarse t(r)")

            intermediate_coarse_t = np.zeros_like(self.data_vv.t)
            intermediate_coarse_t[:node_index[-1], ...] = self.data_vv.t[node_index[-1], ...]
            new_coarse_t = self.step_Picard(intermediate_coarse_t, previous_coarse_t)

            
            if np.absolute((previous_coarse_t - new_coarse_t)).max() < 1e-5:
                print("Iteration complete")
                print("Diff: {diff}".format(diff=np.absolute((previous_coarse_t - new_coarse_t)).max()))
                break
            else:
                print("Diff: {diff} (not below tolerance)\n".format(diff=np.absolute((previous_coarse_t - new_coarse_t)).max()))
                coarse_t = new_coarse_t


            

    def solve_uv(self, RISM, Closure, lam):
        pass

    def step_NR(self, Q, P, A_curr, A_prev):
        ck = np.zeros_like(self.data_vv.c)

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            ck[..., i, j] = self.data_vv.grid.dht(self.data_vv.c[..., i, j])

        return step_NR(self.nbasis, 
                        self.data_vv.ns1, 
                        self.data_vv.ns2, 
                        self.data_vv.grid.npts, 
                        self.data_vv.w, ck, 
                        self.data_vv.p, 
                        self.data_vv.grid.ki, self.data_vv.grid.ri,
                        self.data_vv.grid.d_k, 
                        self.data_vv.grid.d_r, 
                        self.data_vv.B, self.data_vv.u_sr, self.data_vv.t, Q, P, A_curr, A_prev)


@njit
def D_calc(ns1, ns2, npts, w, ck, p, k_grid, r_grid, dk, l):

    D = np.zeros((npts, ns1, ns2, ns1, ns2), dtype=np.float64)
    coskr = np.zeros((npts), dtype=np.float64)
    M1 = np.zeros((ns1, ns2), dtype=np.float64)
    M2 = np.zeros_like(M1)
    I = np.identity(ns1, dtype=np.float64)
    M1 = np.linalg.inv((I - w[l] @ ck[l] @ p)) @ w[l]
    M2 = np.linalg.inv((I - w[l] @ ck[l] @ p)) @ w[l]
    for m in prange(npts):
        coskr[m] = np.cos(k_grid[m] * r_grid[l])
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
                    D[:, i, j, k, l] = coskr[:] * (M1[i, j] * M2[k, l] - kron1 * kron2)

    return D.sum(axis=0) * dk

@njit
def E_calc(ns1, ns2, npts, w, ck, p, k_grid, r_grid, dk, l):

    E = np.zeros((npts, ns1, ns2, ns1, ns2), dtype=np.float64)
    sinkr = np.zeros((npts), dtype=np.float64)
    M1 = np.zeros((ns1, ns2), dtype=np.float64)
    M2 = np.zeros_like(M1)
    I = np.identity(ns1, dtype=np.float64)

    M1 = np.linalg.inv((I - w[l] @ ck[l] @ p)) @ w[l]
    M2 = np.linalg.inv((I - w[l] @ ck[l] @ p)) @ w[l]
    for m in prange(npts):
        sinkr[m] = k_grid[m] * np.sin(k_grid[m] * r_grid[l])

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
                    # print(i, j, k, l)
                    D[:, i, j, k, l] = coskr[:] * (
                        M1[:, i, j] * M2[:, k, l] - kron1 * kron2
                    )

    return D.sum(axis=0) * dk
