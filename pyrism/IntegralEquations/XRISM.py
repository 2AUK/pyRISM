import numpy as np
from Core import RISM_Obj


def XRISM(data):
    """
    Computes RISM equation in the form:

    h = w*c*(wcp)^-1*w*c

    Parameters
    ----------
    wk: Intramolecular correlation function 3D-array

    cr: Direct correlation function 3D-array

    vrlr: Long-range potential

    rho: Number density matrix

    Returns
    -------
    trsr: ndarray
    An array containing short-range indirection correlation function
    """
    I = np.eye(data.ns1, M=data.ns2)
    ck = np.zeros((data.npts, data.ns1, data.ns2), dtype=np.float64)
    for i, j in np.ndindex(data.ns1, data.ns2):
        ck[:, i, j] = data.grid.dht(data.c[:, i, j])
        ck[:, i, j] -= data.B * data.uk_lr[:, i, j]
    for i in range(data.grid.npts):
        iwcp = np.linalg.inv(I - data.w[i, :, :] @ ck[i, :, :] @ data.p)
        wcw = data.w[i, :, :] @ ck[i, :, :] @ data.w[i, :, :]
        data.h[i, :, :] = iwcp @ wcw - ck[i, :, :]
    for i, j in np.ndindex(data.ns1, data.ns2):
        data.t[:, i, j] = data.grid.idht(data.h[:, i, j] - (
            data.B * data.uk_lr[:, i, j]))
