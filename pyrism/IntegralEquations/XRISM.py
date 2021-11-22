import numpy as np
from Core import RISM_Obj
from dataclasses import dataclass, field

@dataclass
class XRISM(object):

    data_vv: RISM_Obj
    data_uv: RISM_Obj = None

    def compute_vv(self):
        I = np.eye(self.data_vv.ns1, M=self.data_vv.ns2)
        ck = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)
        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            ck[:, i, j] = self.data_vv.grid.dht(self.data_vv.c[:, i, j])
            ck[:, i, j] -= self.data_vv.B * self.data_vv.uk_lr[:, i, j]
        for i in range(self.data_vv.grid.npts):
            iwcp = np.linalg.inv(I - self.data_vv.w[i] @ ck[i] @ self.data_vv.p)
            wcw = self.data_vv.w[i] @ ck[i] @ self.data_vv.w[i]
            self.data_vv.h[i] = iwcp @ wcw
        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.t[:, i, j] = self.data_vv.grid.idht(self.data_vv.h[:, i, j] - ck[:, i, j]) - (
                self.data_vv.B * self.data_vv.ur_lr[:, i, j])

    def compute_uv(self):
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
