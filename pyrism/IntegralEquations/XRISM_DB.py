#!/usr/bin/env python3

import numpy as np
from pyrism.Core import RISM_Obj
from dataclasses import dataclass, field
from numba import jit, njit, prange


@dataclass
class XRISM_DB(object):
    eps: float = field(init=False)
    b_param: float
    data_vv: RISM_Obj
    data_uv: RISM_Obj = None

    def _pot_builder(self, dat1, dat2):
        for i, iat in enumerate(dat1.atoms):
            for j, jat in enumerate(dat2.atoms):
                i_sr_params = iat.params[:-1]
                j_sr_params = jat.params[:-1]
                qi = iat.params[-1]
                qj = jat.params[-1]
                dat1.ur_lr[:, i, j] = self._coulomb_r(dat2.grid.ri, qi, qj, dat2.amph)
                dat1.uk_lr[:, i, j] = self._coulomb_k(dat2.grid.ki, qi, qj, dat2.amph)

    def _coulomb_r(self, r, q1, q2, charge_coeff):
        return charge_coeff * q1 * q2 / r

    def _coulomb_k(self, k, q1, q2, charge_coeff):
        return charge_coeff * 4.0 * np.pi * q1 * q2 / np.power(k, 2)

    def construct_Q(self, dat1):
        I = np.eye(dat1.ns1, M=dat1.ns2)
        u = -dat1.B * dat1.uk_lr
        for i in range(dat1.grid.npts):
            wu = dat1.w[i] @ u[i] @ dat1.w[i]
            iwu = np.linalg.inv(I - dat1.w[i] @ u[i])
            dat1.Q_k[i] = wu @ iwu @ dat1.w[i]

    def mu_sq(self, dat1):
        mu = 0.0
        for isp in dat1.species:
            for iat in isp.atom_sites:
                mu += isp.dens * iat.params[-1] * iat.params[-1]
        return 4.0 * np.pi * dat1.B * mu

    def compute_vv(self):
        ur_lr = -self.data_vv.B * self.data_vv.ur_lr
        uk_lr = -self.data_vv.B * self.data_vv.uk_lr

        self._pot_builder(self.data_vv, self.data_vv)
        self.data_vv.u_sr = self.data_vv.u - self.data_vv.ur_lr
        self.construct_Q(self.data_vv)
        mu_sq = self.mu_sq(self.data_vv)

        Q_lr_k = uk_lr / (
            1.0 + (mu_sq / np.power(self.data_vv.grid.ki[:, np.newaxis, np.newaxis], 2))
        )

        Q_lr_r = ur_lr * np.exp(
            -np.sqrt(mu_sq) * self.data_vv.grid.ri[:, np.newaxis, np.newaxis]
        )

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.Q_r[..., i, j] = (
                self.data_vv.grid.idht(self.data_vv.Q_k[..., i, j] - Q_lr_k[..., i, j])
                + Q_lr_r[..., i, j]
            )
            # self.data_vv.Q_r[..., i, j] = self.data_vv.grid.idht(self.data_vv.Q_k[..., i, j])

        ck = np.zeros(
            (self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64
        )

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            ck[..., i, j] = self.data_vv.grid.dht(self.data_vv.c[..., i, j])

        self.data_vv.h = vv_impl(
            self.data_vv.ns1,
            self.data_vv.ns2,
            self.data_vv.npts,
            ck,
            self.data_vv.B,
            self.data_vv.uk_lr,
            self.data_vv.w,
            self.data_vv.p,
            self.data_vv.Q_k,
        )

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.t[:, i, j] = self.data_vv.grid.idht(
                self.data_vv.h[:, i, j] - ck[:, i, j]
            )
        self.data_vv.tau = self.data_vv.t - self.data_vv.Q_r + ur_lr

    def compute_uv(self):
        if self.data_uv is not None:
            ck_uv = np.zeros(
                (self.data_uv.npts, self.data_uv.ns1, self.data_uv.ns2),
                dtype=np.float64,
            )

            for i, j in np.ndindex(self.data_uv.ns1, self.data_uv.ns2):
                ck_uv[..., i, j] = self.data_uv.grid.dht(self.data_uv.c[..., i, j])

            self.data_uv.h = uv_impl(
                self.data_uv.ns1,
                self.data_uv.ns2,
                self.data_uv.npts,
                ck_uv,
                self.data_uv.B,
                self.data_uv.uk_lr,
                self.data_uv.w,
                self.data_vv.w,
                self.data_uv.p,
                self.data_vv.h,
            )

            for i, j in np.ndindex(self.data_uv.ns1, self.data_uv.ns2):
                self.data_uv.t[:, i, j] = (
                    self.data_uv.grid.idht(self.data_uv.h[:, i, j] - ck_uv[:, i, j])
                    - self.data_uv.B * self.data_uv.ur_lr[:, i, j]
                )
        else:
            raise RuntimeError("uv dataclass not defined")


@njit(parallel=True)
def vv_impl(ns1, ns2, npts, ck, B, uk_lr, w, p, Q):
    I = np.eye(ns1, M=ns2)
    h = np.zeros((npts, ns1, ns2), dtype=np.float64)

    w += Q
    for i in prange(npts):
        iwcp = np.linalg.inv(I - w[i] @ ck[i] @ p)
        wc = w[i] @ ck[i]
        h[i] = wc @ iwcp @ w[i]

    return h + Q


@njit(parallel=True)
def uv_impl(ns1, ns2, npts, ck_uv, B, uk_lr, w_uv, w_vv, p, h_vv):
    I = np.eye(ns1, M=ns2)
    h_uv = np.zeros((npts, ns1, ns2), dtype=np.float64)

    for i in prange(npts):
        h_uv[i] = (w_uv[i] @ ck_uv[i]) @ (w_vv[i] + p @ h_vv[i])

    return h_uv
