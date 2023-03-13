import numpy as np
from pyrism.Core import RISM_Obj


def Gaussian_Fluctuations(data, vv=None):
    mu = -4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                               ((0.5 * data.c * data.h) + data.c) @ data.p[np.newaxis, ...])
    return np.sum(mu, axis=(1, 2)) / data.B * data.kU

def HyperNetted_Chain(data, vv=None):
    mu = 4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                              ((0.5 * data.t * data.h) - data.c) @ data.p[np.newaxis, ...])
    return np.sum(mu, axis=(1, 2)) / data.B * data.kU

def Kovalenko_Hirata(data, vv=None):
    mu = 4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                              (0.5 * np.power(data.h, 2) * np.heaviside(-data.h, 1) -
                                               (0.5 * data.h * data.c) - data.c) @ data.p[np.newaxis, ...])
    return np.sum(mu, axis=(1, 2)) / data.B * data.kU

def Partial_Wave(data, vv=None):
    h = np.zeros_like(data.h)

    for i, j in np.ndindex(data.ns1, data.ns2):
        h[..., i, j] = data.grid.dht(data.h[..., i, j])

    h_bar = np.zeros_like(h)
    h_bar_k = np.zeros_like(h)
    w_v = vv.w
    w_u = data.w

    for i in range(data.grid.npts):
        h_bar_k[i, ...] = np.linalg.inv(w_u[i, ...]) @ h[i, ...] @ np.linalg.inv(w_v[i, ...])

    for i, j in np.ndindex(data.ns1, data.ns2):
        h_bar[..., i, j] = data.grid.idht(h_bar_k[..., i, j])

    mu = -4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                               ((0.5 * data.c * data.h)
                                                + data.c
                                                - (0.5 * h_bar * data.h)) @ data.p[np.newaxis, ...])
    return np.sum(mu, axis=(1, 2)) / data.B * data.kU
