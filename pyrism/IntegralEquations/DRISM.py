#!/usr/bin/env python3

import numpy as np
from Core import RISM_Obj
from dataclasses import dataclass, field
import Util
from scipy.special import spherical_jn

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
        I = np.eye(self.data_vv.ns1, M=self.data_vv.ns2)
        ck = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)
        w_bar = np.zeros((self.data_vv.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float64)
        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            ck[:, i, j] = self.data_vv.grid.dht(self.data_vv.c[:, i, j])
            ck[:, i, j] -= self.data_vv.B * self.data_vv.uk_lr[:, i, j]
        for i in range(self.data_vv.grid.npts):
            w_bar[i] = self.data_vv.w[i] + self.chi[i]
            iwcp = np.linalg.inv(I - w_bar[i] @ ck[i] @ self.data_vv.p)
            wcw = w_bar[i] @ ck[i] @ w_bar[i]
            self.data_vv.h[i] = iwcp @ wcw + self.chi[i]
        for i, j in np.ndindex(self.data_vv.ns1, self.data_vv.ns2):
            self.data_vv.t[:, i, j] = self.data_vv.grid.idht(self.data_vv.h[:, i, j] - ck[:, i, j]) - (
                self.data_vv.B * self.data_vv.ur_lr[:, i, j])

    def compute_uv(self):
        pass

    def calculate_DRISM_params(self):
        total_density = 0
        dm, _ = Util.dipole_moment(self.data_vv)
        for isp in self.data_vv.species:
            total_density += isp.dens
        dmdensity = total_density * dm * dm
        ptxv = self.data_vv.species[0].dens / total_density
        self.y = 4.0 * np.pi * self.data_vv.B * dmdensity / 9.0
        self.h_c0 = (((self.diel - 1.0) / self.y / (total_density * ptxv)) - 3.0)

    def D_matrix(self):

        d0x = np.zeros((self.data_vv.ns1), dtype=np.float)
        d0y = np.zeros((self.data_vv.ns1), dtype=np.float)
        d1z = np.zeros((self.data_vv.ns1), dtype=np.float)
        for ki, k in enumerate(self.data_vv.grid.ki):
            hck = self.h_c0 * np.exp(-(self.adbcor * k * k))
            for isp in self.data_vv.species:
                for iat in isp.atom_sites:
                    for i in np.ndindex((self.data_vv.ns1)):
                        k_coord = k*iat.coords
                        if k_coord[0] == 0:
                            d0x[i] = 1.0
                        else:
                            d0x[i] = spherical_jn(0, k_coord[0])
                        if k_coord[1] == 0:
                            d0y[i] = 1.0
                        else:
                            d0y[i] = spherical_jn(0, k_coord[1])
                        if k_coord[2] == 0:
                            d1z[i] = 1.0
                        else:
                            d1z[i] = spherical_jn(1, k_coord[2])
            for i, j in np.ndindex((self.data_vv.ns1, self.data_vv.ns2)):
                self.chi[ki, i, j] = d0x[i] * d0y[i] * d1z[i] * hck * d0x[j] * d0y[j] * d1z[j]

    def __post_init__(self):
        self.calculate_DRISM_params()
        self.chi = np.zeros((self.data_vv.grid.npts, self.data_vv.ns1, self.data_vv.ns2), dtype=np.float)
        self.D_matrix()

def vv_impl():
    pass

def uv_impl():
    pass
