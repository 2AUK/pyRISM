import numpy as np

def geometric_mean(A, B):
    return np.sqrt(A * B)

def arithmetic_mean(A, B):
    return 0.5 * (A + B)

def Lorentz_Berthelot(self, eps1: float, eps2: float, sig1: float, sig2: float) -> tuple:
    """
    Lorentz-Berthelot mixing rules to compute epsilon and sigma parameters of different site types

    Parameters
    ----------

    eps1 : float
        epsilon parameter of site 1

    eps2 : float
        epsilon parameter of site 2

    sig1 : float
        sigma parameter of site 1

    sig2 : float
        sigma parameter of site 2

    Returns
    -------
    eps : float
    Mixed epsilon parameter
    sig : float
    Mixed sigma parameter
    """
    eps = geometric_mean(eps1, eps2)
    sig = arithmetic_mean(sig1, sig2)
    return eps, sig

def AB_mixing(A_1, A_2, B_1, B_2):
    
    A = geometric_mean(A_1, A_2)
    B = geometric_mean(B_1, B_2)
    return A, B