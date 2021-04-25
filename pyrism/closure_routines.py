import numpy as np


def HyperNetted_Chain(vr, tr):
    return np.exp(-vr + tr) - 1.0 - tr


def KovalenkoHirata(vr, tr):
    return np.where((-vr + tr) <= 0, np.exp(-vr + tr) - 1.0 - tr, -vr)


def KobrynGusarovKovalenko(vr, tr):
    zeros = np.zeros_like(tr)
    return np.maximum(zeros, -vr)


def PercusYevick(vr, tr):
    return np.exp(-vr) * (1.0 + tr) - tr - 1.0


def PSE_n(vr, tr, n):
    t_fac = 0
    for i in range(n):
        t_fac += np.power((-vr + tr), i) / np.math.factorial(i)
    return np.where((-vr + tr) <= 0, np.exp(-vr + tr) - 1.0 - tr, t_fac - 1.0 - tr)


def PSE_1(vr, tr, n):
    t_fac = 0
    for i in range(n):
        t_fac += np.power((-vr + tr), i) / np.math.factorial(i)
    return np.where((-vr + tr) <= 0, np.exp(-vr + tr) - 1.0 - tr, t_fac - 1.0 - tr)


def PSE_2(vr, tr, n):
    t_fac = 0
    for i in range(n):
        t_fac += np.power((-vr + tr), i) / np.math.factorial(i)
    return np.where((-vr + tr) <= 0, np.exp(-vr + tr) - 1.0 - tr, t_fac - 1.0 - tr)


def PSE_3(vr, tr, n):
    t_fac = 0
    for i in range(n):
        t_fac += np.power((-vr + tr), i) / np.math.factorial(i)
    return np.where((-vr + tr) <= 0, np.exp(-vr + tr) - 1.0 - tr, t_fac - 1.0 - tr)