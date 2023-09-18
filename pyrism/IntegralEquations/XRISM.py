import numpy as np
from pyrism import Util
from pyrism.Core import RISM_Obj
from dataclasses import dataclass, field
from numba import jit, njit, prange

@dataclass
class XRISM(object):

    data_vv: RISM_Obj
    data_uv: RISM_Obj = None

    # def compute_vv(self):

    #     self.data_vv.h_k, self.data_vv.t = xrism(
    #         self.data_vv.ns1,
    #         self.data_vv.npts,
    #         self.data_vv.grid.ri,
    #         self.data_vv.grid.ki,
    #         self.data_vv.grid.d_r,
    #         self.data_vv.grid.d_k,
    #         self.data_vv.c,
    #         self.data_vv.w,
    #         self.data_vv.p,
    #         self.data_vv.B,
    #         self.data_vv.uk_lr,
    #         self.data_vv.ur_lr
    #     )

    def compute_vv(self):

        ck = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)

        ck = self.data_vv.grid.dht(self.data_vv.c)
        
        self.data_vv.h = vv_impl(self.data_vv.ns1,
                                 self.data_vv.ns2,
                                 self.data_vv.npts,
                                 ck,
                                 self.data_vv.B,
                                 self.data_vv.uk_lr,
                                 self.data_vv.w,
                                 self.data_vv.p)

        
        self.data_vv.t = self.data_vv.grid.idht(self.data_vv.h - ck) \
                - self.data_vv.B * self.data_vv.ur_lr

        self.data_vv.h_k = self.data_vv.h

    def compute_uv(self):
        if self.data_uv is not None:

            ck_uv = np.zeros((self.data_uv.npts, self.data_uv.ns1, self.data_uv.ns2), dtype=np.float64)

            ck_uv = self.data_uv.grid.dht(self.data_uv.c)

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

            self.data_uv.t = self.data_uv.grid.idht(self.data_uv.h - ck_uv) \
                    - self.data_uv.B * self.data_uv.ur_lr

            self.data_uv.h_k = self.data_uv.h
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
