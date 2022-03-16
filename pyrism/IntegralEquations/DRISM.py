#!/usr/bin/env python3

import numpy as np
from Core import RISM_Obj
from dataclasses import dataclass, field
import Util
from scipy.special import spherical_jn
from numba import njit, prange

@dataclass
class DRISM(object):

    data_vv: RISM_Obj
    diel: float
    adbcor: float
    data_uv: RISM_Obj = None
    chi: np.ndarray = field(init=False)
    h_c0: float = field(init=False)
    y: float = field(init=False)

    def compute_vv(self):

        ck = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            ck[..., i, j] = self.data_vv.grid.dht(self.data_vv.c[..., i, j])

        self.data_vv.h = vv_impl(self.data_vv.ns1,
                                 self.data_vv.ns2,
                                 self.data_vv.npts,
                                 ck,
                                 self.data_vv.B,
                                 self.data_vv.uk_lr,
                                 self.data_vv.w,
                                 self.data_vv.p,
                                 self.chi)

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.t[:, i, j] = self.data_vv.grid.idht(self.data_vv.h[:, i, j] - ck[:, i, j]) \
                - self.data_vv.B * self.data_vv.ur_lr[:, i, j]

        """
        I = np.eye(self.data_vv.ns1, M=self.data_vv.ns2, dtype=np.float64)
        ck = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)
        w_bar = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)
        k = self.data_vv.grid.ki
        r = self.data_vv.grid.ri
        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            ck[:, i, j] = self.data_vv.grid.dht(self.data_vv.c[:, i, j])
            ck[:, i, j] -= self.data_vv.B * self.data_vv.uk_lr[:, i, j]
        for i in range(self.data_vv.grid.npts):
            chi = self.chi
            w_bar[i] = (self.data_vv.w[i] + self.data_vv.p @ chi[i])
            iwcp = np.linalg.inv(I - w_bar[i] @ ck[i] @ self.data_vv.p)
            wcw = (w_bar[i] @ ck[i] @ w_bar[i])
            self.data_vv.h[i] = (iwcp @ wcw) + (chi[i])
        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.t[:, i, j] = self.data_vv.grid.idht(self.data_vv.h[:, i, j] - ck[:, i, j]) - (
                self.data_vv.B * self.data_vv.ur_lr[:, i, j])
        """
    def compute_uv(self):
        if self.data_uv is not None:

            ck_uv = np.zeros((self.data_uv.npts, self.data_uv.ns1, self.data_uv.ns2), dtype=np.float64)

            for i, j in np.ndindex(self.data_uv.ns1, self.data_uv.ns2):
                ck_uv[..., i, j] = self.data_uv.grid.dht(self.data_uv.c[..., i, j])

            self.data_uv.h = uv_impl(self.data_uv.ns1,
                                     self.data_uv.ns2,
                                     self.data_uv.npts,
                                     ck_uv,
                                     self.data_uv.B,
                                     self.data_uv.uk_lr,
                                     self.data_uv.w,
                                     self.data_vv.w,
                                     self.data_uv.p,
                                     self.data_vv.h)

            for i, j in np.ndindex(self.data_uv.ns1, self.data_uv.ns2):
                self.data_uv.t[:, i, j] = self.data_uv.grid.idht(self.data_uv.h[:, i, j] - ck_uv[:, i, j]) \
                    - self.data_uv.B * self.data_uv.ur_lr[:, i, j]
        else:
            raise RuntimeError("uv dataclass not defined")
        """
        if self.data_uv is not None:
            I = np.eye(self.data_uv.ns1, M=self.data_uv.ns2)
            ck_uv = np.zeros((self.data_uv.npts, self.data_uv.ns1, self.data_uv.ns2), dtype=np.float64)
            for i, j in np.ndindex(self.data_uv.ns1, self.data_uv.ns2):
                ck_uv[:, i, j] = self.data_uv.grid.dht(self.data_uv.c[:, i, j])
                ck_uv[:, i, j] -= self.data_uv.B * self.data_uv.uk_lr[:, i, j]
            for i in range(self.data_uv.grid.npts):
                self.data_uv.h[i] = (self.data_uv.w[i] @ ck_uv[i]) @ (self.data_vv.w[i] + self.data_vv.p @ self.data_vv.h[i])
            for i, j in np.ndindex(self.data_uv.ns1, self.data_uv.ns2):
                self.data_uv.t[:, i, j] = self.data_uv.grid.idht(self.data_uv.h[:, i, j] - ck_uv[:, i, j]) - (
                    self.data_uv.B * self.data_uv.ur_lr[:, i, j])
        else:
            raise RuntimeError("uv dataclass not defined")
        """

    def calculate_DRISM_params(self):
        total_density = 0
        Util.align_dipole(self.data_vv)
        dm, _ = Util.dipole_moment(self.data_vv)
        for isp in self.data_vv.species:
            total_density += isp.dens
        dmdensity = total_density * dm * dm
        ptxv = self.data_vv.species[0].dens / total_density
        self.y = 4.0 * np.pi * dmdensity / 9.0
        self.h_c0 = (((self.diel - 1.0) / self.y) - 3.0) / (total_density * ptxv)

    def D_matrix(self):

        d0x = np.zeros((self.data_vv.ns1), dtype=np.float)
        d0y = np.zeros((self.data_vv.ns1), dtype=np.float)
        d1z = np.zeros((self.data_vv.ns1), dtype=np.float)
        for ki, k in enumerate(self.data_vv.grid.ki):
            hck = self.h_c0 * np.exp(-np.power((self.adbcor * k / 2.0), 2))
            i = -1
            for isp in self.data_vv.species:
                for iat in isp.atom_sites:
                    i += 1
                    k_coord = k*iat.coords
                    if k_coord[0] == 0.0:
                        d0x[i] = 1.0
                    else:
                        d0x[i] = Util.j0(k_coord[0])
                    if k_coord[1] == 0.0:
                        d0y[i] = 1.0
                    else:
                        d0y[i] = Util.j0(k_coord[1])
                    if k_coord[2] == 0.0:
                        d1z[i] = 0.0
                    else:
                        d1z[i] = Util.j1(k_coord[2])
            for i, j in np.ndindex((self.data_vv.ns1, self.data_vv.ns2)):
                self.chi[ki, i, j] = d0x[i] * d0y[i] * d1z[i] * hck * d0x[j] * d0y[j] * d1z[j]

    def __post_init__(self):
        self.calculate_DRISM_params()
        self.chi = np.zeros((self.data_vv.grid.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float)
        self.D_matrix()

@njit(parallel=True)
def vv_impl(ns1, ns2, npts, ck, B, uk_lr, w, p, chi):

    I = np.eye(ns1, M=ns2, dtype=np.float64)
    w_bar = np.zeros((npts, ns1, ns2), dtype=np.float64)
    h = np.zeros_like(w_bar)

    ck -= B * uk_lr
    for i in prange(npts):
        w_bar[i] = (w[i] + p @ chi[i])
        iwcp = np.linalg.inv(I - w_bar[i] @ ck[i] @ p)
        wcw = w_bar[i] @ ck[i] @ w_bar[i]
        h[i] = (iwcp @ wcw) + chi[i]

    return h

@njit(parallel=True)
def uv_impl(ns1, ns2, npts, ck_uv, B, uk_lr, w_uv, w_vv, p, h_vv):
    I = np.eye(ns1, M=ns2)
    h_uv = np.zeros((npts, ns1, ns2), dtype=np.float64)

    ck_uv -= B * uk_lr
    for i in prange(npts):
        h_uv[i] = (w_uv[i] @ ck_uv[i]) @ (w_vv[i] + p @ h_vv[i])

    return h_uv
