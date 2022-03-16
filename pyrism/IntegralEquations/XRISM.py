import numpy as np
import Util
from Core import RISM_Obj
from dataclasses import dataclass, field
from numba import jit, njit, prange
from Core import discrete_hankel_transform, inverse_discrete_hankel_transform

@dataclass
class XRISM(object):

    data_vv: RISM_Obj
    data_uv: RISM_Obj = None

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
                                 self.data_vv.p)

        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.t[:, i, j] = self.data_vv.grid.idht(self.data_vv.h[:, i, j] - ck[:, i, j]) \
                - self.data_vv.B * self.data_vv.ur_lr[:, i, j]

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

@njit(parallel=True)
def vv_impl(ns1, ns2, npts, ck, B, uk_lr, w, p):
    I = np.eye(ns1, M=ns2)
    h = np.zeros((npts, ns1, ns2), dtype=np.float64)

    ck -= B * uk_lr
    for i in prange(npts):
        iwcp = np.linalg.inv(I - w[i] @ ck[i] @ p)
        wcw = w[i] @ ck[i] @ w[i]
        h[i] = iwcp @ wcw

    return h

@njit(parallel=True)
def uv_impl(ns1, ns2, npts, ck_uv, B, uk_lr, w_uv, w_vv, p, h_vv):
    I = np.eye(ns1, M=ns2)
    h_uv = np.zeros((npts, ns1, ns2), dtype=np.float64)

    ck_uv -= B * uk_lr
    for i in prange(npts):
        h_uv[i] = (w_uv[i] @ ck_uv[i]) @ (w_vv[i] + p @ h_vv[i])

    return h_uv
