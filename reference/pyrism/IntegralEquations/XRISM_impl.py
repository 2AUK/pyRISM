#!/usr/bin/env python3

import numpy as np


# pythran export vv_impl(int, int, float, float64 [], float, float64 [], float64 [], float64 [])
def vv_impl(ns1, ns2, npts, ck, B, uk_lr, w, p):
    I = np.eye(ns1, M=ns2)
    h = np.zeros((npts, ns1, ns2), dtype=np.float64)

    ck -= B * uk_lr
    for i in range(npts):
        iwcp = np.linalg.inv(I - w[i] @ ck[i] @ p)
        wcw = w[i] @ ck[i] @ w[i]
        h[i] = iwcp @ wcw

    return h
