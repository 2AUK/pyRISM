"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Handles the primary functions
"""
import numpy as np

#Solvent Info#

NSV = 3
# Site = [Site1, Site2, ... , SiteN]
# SiteN = [Atomic symbol, eps, sig, charge, rho]
Solvent_Sites = [["O", 1.08E-21, 3.166, -0.82, 0.3334], 
                ["H", 3.79E-22, 1.0, 0.41, 0.3334], 
                ["H", 3.79E-22, 1.0, 0.41, 0.3334]]

Solvent_Distances = np.asarray([[0.0, 1.0, 1.0],
                                [1.0, 1.633, 1.633],
                                [1.0, 1.633, 1.633]])

#Solute Info#

Solute_Sites = [] #Nothing for now

class RISM_CONTROLLER(object):
    
    def __init__(self, nsv: int, nsu: int, npts: float, radius: float):
        self.nsu = nsu
        self.nsv = nsv
        self.npts = npts
        self.dr = (radius / (npts))
        self.dk = (2*np.pi / (2*self.npts*self.dr))
        self.radius = radius

def computeLJpot(eps: float, sig: float, r: float) -> float:
    """
    Computes the Lennard-Jones potential

    Parameters
    ----------

    eps : float
        Epsilon parameter used for LJ equation
    sig : float
        Sigma parameter used for LJ equation
    r   : float
        In the context of rism, r corresponds to grid points upon which
        RISM equations are solved

    Returns
    -------
    result : float
        The result of the LJ computation
    """
    return 4.0 * eps * ((sig/r)**12 - (sig/r)**6)

def RhoMat(site_list: list) -> 'ndarray':
    """
    Creates a matrix for the number density of a set of sites for molecules

    Parameters
    ----------

    site_list : list
        A list of lists wherein each list contains information for each site of the
        molecules.

    Returns
    -------
    rho_mat : ndarray
        An array with number densities of each site down the diagonal
    """
    return np.diag([site[-1] for site in site_list])

def CalcWk(dr: float, dk: float, npts: float, site_list: list, l_vv : 'ndarray') -> 'ndarray':
    nsites = len(site_list)
    wk = np.zeros((nsites, nsites, npts), dtype=float)
    for i in range(0, nsites):
        for j in range(0, nsites):
            for l in range(0, npts):
                if (i == j):
                    wk[i][j][l] = 1.0
                else:
                    wk[i][j][l] = np.sin(dk * (l+.5) * l_vv[i][j]) / (dk * (l+.5) * l_vv[i][j])
    return wk
                



if __name__ == "__main__":
    rismobj = RISM_CONTROLLER(3, 0, 4.0, 20.48)
    print(CalcWk(rismobj.dr, rismobj.dk, 4, Solvent_Sites, Solvent_Distances))