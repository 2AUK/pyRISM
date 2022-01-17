import numpy as np
from Core import RISM_Obj


def Gaussian_Fluctuations(data):
    mu = -4.0 * np.pi * data.grid.d_r * np.sum(np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                               ((0.5 * data.c * data.h) + data.c) @ data.p[np.newaxis, ...])
    return mu / data.B * data.kU

def HyperNetted_Chain(data):
    mu = 4.0 * np.pi * data.grid.d_r * np.sum(np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                              ((0.5 * data.t * data.h) - data.c) @ data.p[np.newaxis, ...])
    return mu / data.B * data.kU

def Kovalenko_Hirata(data):
    mu = 4.0 * np.pi * data.grid.d_r * np.sum(np.power(data.grid.ri, 2)[:, np.newaxis, np.newaxis] *
                                              (0.5 * np.power(data.h, 2 * np.heaviside(-data.h, 1)) -
                                              (0.5 * data.h * data.c) - data.c) @ data.p[np.newaxis, ...])
    return mu / data.B * data.kU
