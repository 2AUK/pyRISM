"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Handles the primary functions
"""
import numpy as np
from scipy.fftpack import dst, idst
from scipy.special import erf
import matplotlib.pyplot as plt

#Solvent Info#

#For now we assume only one solvent is being used within the site_list object
NSV = 3
k_b = 1.380649E-23
N_A = 6.02214086E23
cmtoA = 1.0E-24
dmtoA = 1.0E-27
mtoA = 1.0E-30
ar_den = (N_A/39.948) * cmtoA * 1.394
Temp = 298.15
beta = (1.0 / (k_b/1000 * Temp))
kT = 1 * Temp
print(ar_den)
# Site = [Site1, Site2, ... , SiteN]
# SiteN = [Atomic symbol, eps, sig, charge, rho]
#Argon fluid. Lennard-Jones parameters from Rahman
#Number density computed from equation $N = \frac{\N_A}{M}\rho$ where \rho is the mass density (1.384 g/cm^3)
Ar_fluid = [["Ar", (120.0/1000.0)*k_b*N_A, 3.4, 0, ar_den]]
print((120.0/1000.0)*k_b*N_A)

Ar_dist = np.asarray([0.00])


Solvent_Sites = [["O", (1.08E-21/1000.0)*N_A, 3.166, -0.82, 0.3334],
                 ["H", (3.79E-22/1000.0)*N_A, 1.0, 0.41, 0.3334],
                 ["H", (3.79E-22/1000.0)*N_A, 1.0, 0.41, 0.3334]]

Solvent_Distances = np.asarray([[0.0, 1.0, 1.0],
                                [1.0, 0.0, 1.633],
                                [1.0, 1.633, 0.0]])

#Solute Info#

Solute_Sites = [] #Nothing for now

class RismController:

    def __init__(self, nsv: int, nsu: int, npts: float, radius: float):
        self.nsu = nsu
        self.nsv = nsv
        self.npts = npts
        self.d_r = (radius / (npts))
        self.d_k = (2*np.pi / (2*self.npts*self.d_r))
        self.radius = radius

def compute_LJpot(eps: float, sig: float, r: float) -> float:
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

def rho_mat(site_list: list) -> 'ndarray':
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

def calc_wkvv(d_k: float, npts: float, site_list: list, l_vv: 'ndarray') -> 'ndarray':
    """
    Creates a matrix for the intramolecular correlation matrix of a molecule

    Parameters
    ----------

    d_k : float
        kspace grid spacing parameter to generate full kspace grid

    npts : float
        number of grid points

    site_list : list
        A list of lists wherein each list contains information for each site of the
        molecules.

    l_vv : ndarray
        Array containing the distance constraints of a molecule

    Returns
    -------
    wkvv : ndarray
       An array containing information on intramolecular correlation
    """
    nsites = len(site_list)
    wk = np.zeros((nsites, nsites, int(npts)), dtype=float)
    for i in range(0, nsites):
        for j in range(0, nsites):
            for l in range(0, int(npts)):
                if i == j:
                    wk[i][j][l] = 1.0
                else:
                    wk[i][j][l] = np.sin(d_k * (l+.5) * l_vv[i][j]) / (d_k * (l+.5) * l_vv[i][j])
    return wk

def calc_urss(d_r: float, npts: float, site_list1: list, site_list2: list) -> 'ndarray':
    """
    Creates a matrix for site-site lennard-jones potential over a grid

    Parameters
    ----------

    d_r : float
        rspace grid spacing parameter to generate full rspace grid

    npts : float
        number of grid points

    site_list1 : list
        A list of lists wherein each list contains information for each site of the
        molecules.

    site_list2 : list
        A list of lists wherein each list contains information for each site of the
        molecules.

    Returns
    -------
    urss : ndarray
       An array containing information on intermolecular potential of the lists of sites specified
    """
    ns1 = len(site_list1)
    ns2 = len(site_list2)
    urss = np.zeros((ns1, ns2, int(npts)), dtype=float)
    for i in np.arange(0, ns1):
        for j in np.arange(0, ns2):
            for l in np.arange(0, int(npts)):
                if site_list1[i] == site_list2[j]:
                    urss[i][j][l] = compute_LJpot(site_list1[i][1], site_list1[i][2], d_r * (l + .5))
                else:
                    eps, sig = mixing_rules(site_list1[i][1], site_list2[j][1], site_list1[i][2], site_list2[j][2])
                    urss[i][j][l] = compute_LJpot(eps, sig, d_r * (l + .5))
    return urss

def mixing_rules(eps1: float, eps2: float, sig1: float, sig2: float) -> tuple:
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
    eps = np.sqrt(eps1*eps2)
    sig = (sig1 + sig2) / 2.0
    return eps, sig

def hnc_closure(urss: 'ndarray', trss: 'ndarray') -> 'ndarray':
    return np.exp(-urss*beta  + trss) - trss - 1

def long_range(d_r: float, npts: float, sig_par: float, site_list1: list, site_list2: list) -> 'ndarray':
    ns1 = len(site_list1)
    ns2 = len(site_list2)
    clr = np.zeros((ns1, ns2, int(npts)), dtype=float)
    for i in np.arange(0, ns1):
        for j in np.arange(0, ns2):
            for l in np.arange(0, int(npts)):
                r = d_r * (l + .5)
                clr[i][j][l] = site_list1[i][3] * site_list2[j][3] * beta * ( (1 - np.exp(-sig_par*r)) / r)
    return clr

def ornstein_zernike(d_r: float, d_k: float, npts: float, site_list: list, rho: 'ndarray', wkss: 'ndarray', cr: 'ndarray') -> 'ndarray':
    ns = len(site_list)
    I = np.identity(ns)
    h = np.zeros((ns, ns, int(npts)), dtype=float)
    ck = np.zeros((ns, ns, int(npts)))
    k = np.zeros(int(npts))
    r = np.zeros(int(npts))
    tr = np.zeros((ns, ns, int(npts)))
    for i in np.arange(0, int(npts)):
        k[i] = (i + .5) * d_k
        r[i] = (i + .5) * d_r
    for i, j in np.ndindex(ns, ns):
        ck[i, j] = dst(cr[i, j] * r, type=1) * np.pi * 2 * d_r / k
    for l in np.arange(0, int(npts)):
        h[:, :, l] = np.linalg.inv(I - wkss[:, :, l]@ck[:, :, l]@rho)@wkss[:, :, l]@ck[:, :, l]@wkss[:, :, l]
    for i, j in np.ndindex(ns, ns):
        tr[i, j] = idst((h[i, j] - ck[i, j]) * k, type=1) * d_k / (4*np.pi*np.pi) / r

    return tr

def init_tr(d_r: float, npts: float, sig_par: float, site_list1: list, site_list2: list) -> 'ndarray':
    ns1 = len(site_list1)
    ns2 = len(site_list2)
    tr = np.zeros((ns1, ns2, int(npts)), dtype=float)
    for i in np.arange(0, ns1):
        for j in np.arange(0, ns2):
            for l in np.arange(0, int(npts)):
                r = d_r * (l + .5)
                tr[i][j][l] = site_list1[i][3] * site_list2[j][3] * beta * ( (erf(sig_par*r)) / r)
    return tr

if __name__ == "__main__":
    print("Hello, RISM!")
    npts = 10.0
    radius = 20.48
    iterval = 110
    tol = 1E-7
    d_r = radius / npts
    d_k = (2*np.pi / (2*npts*d_r))
    nsites = len(Solvent_Sites)
    narsites = len(Ar_fluid)
    arwk = calc_wkvv(d_k, npts, Ar_fluid, Ar_dist)
    urvv = calc_urss(d_r, npts, Ar_fluid, Ar_fluid)
    cr = np.zeros((narsites, narsites, int(npts)), dtype=float)
    tr = init_tr(d_r, npts, 1.0, Ar_fluid, Ar_fluid)
    clr = long_range(d_r, npts, 1.0, Ar_fluid, Ar_fluid)
    damp = 0.215
    i = 0
    print(arwk)
    print(urvv)
    print(urvv*beta)
    print(np.exp(-urvv))
    print(np.exp(-urvv*beta))
    while i < iterval:
        print("Iteration: ", i)
        tr_old = tr
        cr = hnc_closure(urvv, tr)
        tr2 = ornstein_zernike(d_r, d_k, npts, Ar_fluid, rho_mat(Ar_fluid), arwk, cr)
        tr_new = (1-damp)*tr_old + damp*tr2
        tr = tr_new
        print(cr)
        i+=1
    tr += clr
    gr = np.exp(-(urvv)/kT + tr)
    r = np.zeros(int(npts))
    for i in np.arange(0, int(npts)):
        r[i] = (i + .5) * d_r
    plt.figure(figsize=[6,6])
    plt.xlim([0.0, radius])
    plt.ylim([-radius/2, radius/2])
    plt.axhline(0, color='grey',linestyle='--',linewidth=2)
    plt.plot(r, gr.flatten())
    plt.show()
