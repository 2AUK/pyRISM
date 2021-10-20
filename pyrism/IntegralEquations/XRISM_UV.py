#!/usr/bin/env python3

import numpy as np
from Core import RISM_Obj

def XRISM_UV(data_vv, data_uv):

    I = np.eye(data_uv.ns1, M=data_uv.ns2)
    ck_uv = np.zeros((data_uv.npts, data_uv.ns1, data_uv.ns2), dtype=np.float64)
    for i, j in np.ndindex(data_uv.ns1, data_uv.ns2):
        ck_uv[:, i, j] = data_uv.grid.dht(data_uv.c[:, i, j])
        ck_uv[:, i, j] -= data_uv.B * data_uv.uk_lr[:, i, j]
    for i in range(data_uv.grid.npts):
        data_uv.h[i] = data_uv.w[i] @ ck_uv[i] @ data_vv.w[i] + (data_uv.w[i] @ ck_uv[i]) @ (data_vv.p @ data_vv.h[i])
    for i, j in np.ndindex(data_uv.ns1, data_uv.ns2):
        data_uv.t[:, i, j] = data_uv.grid.idht(data_uv.h[:, i, j] - ck_uv[:, i, j]) - (
            data_uv.B * data_uv.ur_lr[:, i, j])
