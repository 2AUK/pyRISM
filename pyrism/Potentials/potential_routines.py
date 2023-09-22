import numpy as np
from scipy.special import erf, erfc
from numba import njit


@njit
def Lennard_Jones(r, params, lam):
    """
    Computes the Lennard-Jones potential with epsilon and sigma parameters

    Parameters
    ----------

    eps: float
        Epsilon parameter used for LJ equation
    sig: float
        Sigma parameter used for LJ equation
    grid.ri: ndarray
        In the context of rism, ri corresponds to grid points upon which
        RISM equations are solved
    lam: float
        Lambda parameter to switch on potential

    Returns
    -------
    result: float
        The result of the LJ computation
    """
    eps, sig = params
    return 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6) * lam


def Lennard_Jones_AB(r, C6, C12, lam):
    """
    Computes the Lennard-Jones potential with C6 and C12 parameters

    Parameters
    ----------

    C6: float
        C6 parameter used for LJ equation
    C12: float
        C12 parameter used for LJ equation
    grid.ri: ndarray
        In the context of rism, ri corresponds to grid points upon which
        RISM equations are solved
    lam: float
        Lambda parameter to switch on potential

    Returns
    -------
    result: float
        The result of the LJ computation
    """

    return ((C12 / r**12) - (C6 / r**6)) * lam


def hard_spheres(r, sigma, lam):
    """
    Computes the Lennard-Jones potential with C6 and C12 parameters

    Parameters
    ----------

    C6: float
        C6 parameter used for LJ equation
    C12: float
        C12 parameter used for LJ equation
    grid.ri: ndarray
        In the context of rism, ri corresponds to grid points upon which
        RISM equations are solved
    lam: float
        Lambda parameter to switch on potential

    Returns
    -------
    result: float
        The result of the LJ computation
    """

    return np.where((r >= sigma), 0, np.inf) * lam


@njit
def coulomb(r, q1, q2, lam, charge_coeff):
    """
    Computes the Coulomb potential

    Parameters
    ----------

    q1: float
        Coulomb charge for particle 1
    q2: float
        Coulomb charge for particle 1
    grid.ri: ndarray
        In the context of rism, ri corresponds to grid points upon which
        RISM equations are solved
    lam: float
        Lambda parameter to switch on potential

    Returns
    -------
    result: float
        The result of the LJ computation
    """
    return lam * charge_coeff * q1 * q2 / r


def coulomb_lr_r(r, q1, q2, damping, rscreen, lam, charge_coeff):
    """
    Computes the Ng renorm potential

    Parameters
    ----------

    q1: float
        Coulomb charge for particle 1
    q2: float
        Coulomb charge for particle 1
    grid.ri: ndarray
        In the context of rism, ri corresponds to grid points upon which
        RISM equations are solved
    damping: float
        Damping parameter for erf
    lam: float
        Lambda parameter to switch on potential

    Returns
    -------
    result: float
        The result of the LJ computation
    """
    return lam * charge_coeff * q1 * q2 * erf(damping * r / rscreen) / r

    # return lam * charge_coeff * q1 * q2 * (1.0 - erfc( r / rscreen))


@njit
def coulomb_lr_k(k, q1, q2, damping, lam, charge_coeff):
    """
    Computes the Ng renorm potential

    Parameters
    ----------

    q1: float
        Coulomb charge for particle 1
    q2: float
        Coulomb charge for particle 1
    grid.ri: ndarray
        In the context of rism, ri corresponds to grid points upon which
        RISM equations are solved
    damping: float
        Damping parameter for erf
    lam: float
        Lambda parameter to switch on potential

    Returns
    -------
    result: float
        The result of the LJ computation
    """
    """
    return (
        lam
        * 4.0
        * np.pi
        * q1
        * q2
        * charge_coeff
        * np.exp(-np.power((damping * k / 2.0), 2.0))
        / k
    )
    """
    return (
        lam
        * 4.0
        * np.pi
        * q1
        * q2
        * charge_coeff
        * np.exp(-1.0 * k**2.0 / (4.0 * damping**2.0))
        / k**2.0
    )
