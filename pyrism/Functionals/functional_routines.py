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

def Single_Component(data, vv=None):
    mu = 4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] * data.c @ data.p[np.newaxis, ...])

    return np.sum(mu, axis=(1, 2)) / data.B * data.kU

def Partial_Wave(data, vv=None):
    h = data.h_k
    h_bar_k = np.zeros_like(h)
    w_v = vv.w
    w_u = data.w

    t_vv_sr = vv.t + (vv.B * vv.ur_lr)
    c_uv_sr = data.c - (data.B * data.ur_lr)

    t_k_sr = np.zeros_like(t_vv_sr)
    c_k_sr = np.zeros_like(c_uv_sr)

    for i, j in np.ndindex(data.ns1, data.ns2):
        c_k_sr[:, i, j] = data.grid.dht(c_uv_sr[:, i, j])

    for a, g in np.ndindex(vv.ns1, vv.ns2):
        t_k_sr[:, a, g] = vv.grid.dht(t_vv_sr[:, a, g])

    t_k = t_k_sr - (vv.B * vv.uk_lr)
    c_k = c_k_sr + (data.B * data.uk_lr)
    mu_pw_comp_r = np.zeros_like(h)

    for i in range(data.grid.npts):
        h_bar_k[i, ...] = np.linalg.inv(w_u[i, ...]) @ h[i, ...] @ np.linalg.inv(w_v[i, ...])

    mu_pw_comp_k = 4.0 * np.pi / np.power(2 * np.pi, 3) * (np.power(data.grid.ki, 2)[:, np.newaxis, np.newaxis] *
                                                         (0.5 * h_bar_k * (c_k @ t_k))
                                                         @ data.p[np.newaxis, ...])
    for i, j in np.ndindex(data.ns1, data.ns2):
        mu_pw_comp_r[:, i, j] = data.grid.idht(mu_pw_comp_k[:, i, j])

    return np.sum(mu_pw_comp_k, axis=(1, 2)) / data.B * data.kU
