import numpy as np
import Data


def HyperNetted_Chain(data):
    return np.exp(-(data.B * data.u_sr) + data.t) - 1.0 - data.t


def KovalenkoHirata(data):
    return np.where(
        (-(data.B * data.u_sr) + data.t) <= 0,
        np.exp(-(data.B * data.u_sr) + data.t) - 1.0 - data.t,
        -(data.B * data.u_sr),
    )


def KobrynGusarovKovalenko(data):
    zeros = np.zeros_like(data.t)
    return np.maximum(zeros, -(data.B * data.u_sr))


def PercusYevick(data):
    return np.exp(-(data.B * data.u_sr)) * (1.0 + data.t) - data.t - 1.0


def PSE_n(data, n):
    t_fac = 0
    for i in range(n):
        t_fac += np.power((-(data.B * data.u_sr) + data.t), i) / np.math.factorial(i)
    return np.where(
        (-(data.B * data.u_sr) + data.t) <= 0,
        np.exp(-(data.B * data.u_sr) + data.t) - 1.0 - data.t,
        t_fac - 1.0 - data.t,
    )


def PSE_1(data):
    return PSE_n(data, 1)


def PSE_2(data):
    return PSE_n(data, 2)


def PSE_3(data):
    return PSE_n(data, 3)