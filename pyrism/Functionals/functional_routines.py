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
    h_uv_k = data.h_k
    h_bar_uv_k = np.zeros_like(h_uv_k)

    for i in range(data.grid.npts):
        h_bar_uv_k[i, ...] = np.linalg.inv(data.w[i, ...]) @ h_uv_k[i, ...] @ np.linalg.inv(vv.w[i, ...])

    h_bar_uv_r = np.zeros_like(h_bar_uv_k)

    for i, j in np.ndindex(data.ns1, data.ns2):
        h_bar_uv_r[:, i, j] = data.grid.idht(h_bar_uv_k[:, i, j])

    mu = -4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] * (data.c + 0.5 * data.c * data.h - (0.5 * h_bar_uv_r * data.h)) @ data.p[np.newaxis, ...])

    return np.sum(mu, axis=(1, 2)) / data.B * data.kU

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
        v_k[:, a, g] = data.grid.dht(np.exp(v_repulsive[:, a, g])-1)

    for s, a, v in np.ndindex(data.ns1, vv.ns1, vv.ns2):
        if a != v:
            expBr[:, s, a] *= data.grid.idht(vv.w[:, a, v] * v_k[:, s, v])+1

    #correction for truncation error
    expBr[expBr < 1e-12] = 1e-12    

    #correction for finite domain (see RISM-MOL implementation)
    N_end = int(data.grid.npts - (data.grid.npts/4))
    
    expBr[N_end:, ...] = 1

    mu = 4.0 * np.pi * (np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] * ((data.h + 1) * (expBr - 1)) @ data.p[np.newaxis, ...])

    return np.sum(mu, axis=(1, 2)) / data.B * data.kU
