import numpy as np
import matplotlib.pyplot as plt
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

    c_uv_sr = data.c - (data.B * data.ur_lr)

    w_v_r = np.zeros_like(w_v)
    c_k_sr = np.zeros_like(c_uv_sr)

    for i, j in np.ndindex(data.ns1, data.ns2):
        c_k_sr[:, i, j] = data.grid.dht(c_uv_sr[:, i, j])

    for a, g in np.ndindex(vv.ns1, vv.ns2):
        w_v_r[:, a, g] = vv.grid.dht(w_v[:, a, g])

    eta = vv.p[np.newaxis, ...] @ (w_v_r + vv.p[np.newaxis, ...] @ vv.h) @ vv.c

    eta_k = np.zeros_like(eta)

    for i, j in np.ndindex(vv.ns1, vv.ns2):
        eta_k[:, i, j] = vv.grid.dht(eta[:, i, j])

    c_k = c_k_sr + (data.B * data.uk_lr)

    for i in range(data.grid.npts):
        h_bar_k[i, ...] = np.linalg.inv(w_u[i, ...]) @ h[i, ...] @ np.linalg.inv(w_v[i, ...])

    mu_pw_comp_k = 4.0 * np.pi / np.power(2 * np.pi, 3) * (np.power(data.grid.ki, 2)[:, np.newaxis, np.newaxis] *
                                                         (0.5 * h_bar_k * (c_k @ eta_k))
                                                         @ data.p[np.newaxis, ...])

    mu_pw_comp_r = np.zeros_like(mu_pw_comp_k)

    for i, j in np.ndindex(data.ns1, data.ns2):
        mu_pw_comp_r[:, i, j] = data.grid.idht(mu_pw_comp_k[:, i, j])

    return np.sum(mu_pw_comp_k, axis=(1, 2)) / data.B * data.kU

def Repulsive_Bridge_Correction(data, vv=None):
    u_repulsive = np.zeros_like(data.u)

    for i, iat in enumerate(data.atoms):
        for j, jat in enumerate(vv.atoms):
            eps_i, sig_i = iat.params[:-1]
            eps_j, sig_j = jat.params[:-1]
            if iat is jat:
                u_repulsive[:, i, j] = 4.0 * eps_i * np.power(( sig_i / data.grid.ri ), 12)
            else:
                mixed_eps = np.sqrt(eps_i * eps_j)
                mixed_sig = 0.5 * (sig_i + sig_j)
                u_repulsive[:, i, j] = 4.0 * mixed_eps * np.power(( mixed_sig / data.grid.ri ), 12)
    
    v_repulsive = -data.B * u_repulsive

    expBr = np.ones_like(u_repulsive)

    v_k = np.zeros_like(u_repulsive)

    for a, g in np.ndindex(data.ns1, data.ns2):
        v_k[:, a, g] = data.grid.dht(np.exp(v_repulsive[:, a, g]))

    for s, a, v in np.ndindex(data.ns1, vv.ns1, vv.ns2):
        if a != v:
            expBr[:, s, a] *= data.grid.idht(vv.w[:, a, v] * v_k[:, s, v])

    #correction for truncation error
    expBr[expBr < 1e-12] = 1e-12    

    #correction for finite domain (see RISM-MOL implementation)
    N_end = int(data.grid.npts - (data.grid.npts/4))
    
    expBr[N_end:, ...] = 1

    mu = 4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] * ((data.h + 1) * (expBr - 1)) @ data.p[np.newaxis, ...])

    return np.sum(mu, axis=(1, 2)) / data.B * data.kU
