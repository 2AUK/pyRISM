import numpy as np


def Lennard_Jones(r, eps, sig, lam, beta):
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
    return beta * 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6) * lam


def Lennard_Jones_AB(r, C6, C12, lam, beta):
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

    return beta * ((C12 / r ** 12) - (C6 / r ** 6)) * lam


def hard_spheres(r, sigma, lam, beta):
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

    return beta * np.where((r >= sigma), 0, np.inf) * lam


def coulomb(r, q1, q2, lam, beta, charge_coeff):
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
    return lam * beta * charge_coeff * q1 * q2 / r


def coulomb_lr_r(r, q1, q2, damping, rscreen, lam, beta, charge_coeff):
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
    return lam * beta * charge_coeff * q1 * q2 * erf(damping * r / rscreen) / r


def coulomb_lr_k(k, q1, q2, damping, lam, beta, charge_coeff):
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
    return (
        lam
        * beta
        * 4
        * np.pi
        * q1
        * q2
        * charge_coeff
        * np.exp(-1.0 * k ** 2 / (4.0 * damping ** 2))
        / k ** 2
    )
