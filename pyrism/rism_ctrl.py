"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Handles the primary functions
"""
import numpy as np
from scipy.fft import dst, idst
from scipy.special import erf
import matplotlib.pyplot as plt
import grid

#Solvent Info#

#For now we assume only one solvent is being used within the site_list object
M_PI =  3.14159265358979323846
NSV = 3
k_b = 1.380649E-23
k_b2 = 0.0019872 #kcal/(mol/K)
N_A = 6.02214086E23
cmtoA = 1.0E-24
dmtoA = 1.0E-27
mtoA = 1.0E-30
ar_den = (N_A/39.948) * cmtoA * 1.374
Temp = 85.0
kTkcal = 1.9827E-3
kT = 1
beta = 1 / (kT * Temp)
charge_coeff = 1
# Site = [Site1, Site2, ... , SiteN]
# SiteN = [Atomic symbol, eps, sig, charge, rho]
#Argon fluid. Lennard-Jones parameters from Rahman (already divided by kB)
#Number density computed from equation $N = \frac{\N_A}{M}\rho$ where \rho is the mass density (1.394 g/cm^3)
Ar_fluid = [["Ar", 120, 3.4, 0, 0.0210145]]

Ar_dist = np.asarray([0.00])

#eps - kcal/mol, sig = A, CHARMM TIP3P
water_sites = [["O", 0.152073, 3.15066, -0.834, 0.03334],
                 ["H", 0, 0.4, 0.417, 0.03334],
                 ["H", 0, 0.4, 0.417, 0.03334]]

Solvent_Distances = np.asarray([[0.0, 0.9572, 0.9572],
                                [0.9572, 0.0, 1.5139],
                                [0.9572, 1.5139, 0.0]])

#Solute Info#

Solute_Sites = [] #Nothing for now

class RismController:

    def __init__(self, nsv: int, nsu: int, npts: float, radius: float, solvent_params: list, dists: np.ndarray):
        self.nsu = nsu
        self.nsv = nsv
        self.grid = grid.Grid(npts, radius)
        self.solvent_sites = solvent_params
        self.dists = dists



    def compute_UR_LJ(self, eps, sig, lam) -> np.ndarray:
        """
        Computes the Lennard-Jones potential

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
        return  beta * 4.0 * eps * ((sig / self.grid.ri)**12 - (sig/self.grid.ri)**6) * lam

    def compute_UR_CMB(self, q1, q2, lam):
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
        return  beta * charge_coeff * q1*q2 / self.grid.ri * lam

    def compute_UR_LR(self, q1, q2, damping, rscreen, lam):
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
        return self.compute_UR_CMB(q1, q2, lam) * erf(damping * self.grid.ri / rscreen)

    def mixing_rules(self, eps1: float, eps2: float, sig1: float, sig2: float) -> tuple:
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

    def build_wk(self):
        """
        Creates a matrix for the intramolecular correlation matrix of a molecule

        Parameters
        ----------
        grid.ki: ndarray
           In the context of rism, ki corresponds to grid points upon which
           RISM equations are solved

        dists: ndarray
            Array containing the distance constraints of a molecule

        Returns
        -------
        wk: ndarray
        An array containing information on intramolecular correlation
        """
        wk = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float)
        I = np.ones(self.grid.npts, dtype=np.float)
        for i, j in np.ndindex(self.nsv, self.nsv):
            if i == j:
                wk[:, i, j] = I
            else:
                wk[:, i, j] = np.sin(self.grid.ki * self.dists[i,j]) / (self.grid.ki * self.dists[i,j])
        return wk
    
    def build_Ur(self, lam):
        """
        Creates a matrix for the potential across the grid

        Returns
        -------
        Ur: ndarray
        An array containing total potential across grid
        """
        vv = True
        if vv == True:
            Ur = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float)
            for i,j in np.ndindex(self.nsv, self.nsv):
                if i == j:
                    eps = self.solvent_sites[i][1]
                    sig = self.solvent_sites[i][2]
                    q = self.solvent_sites[i][3]
                    Ur[:, i, j] = self.compute_UR_LJ(eps, sig, lam) + self.compute_UR_CMB(q, q, lam)
                else:
                    eps, sig = self.mixing_rules(self.solvent_sites[i][1],
                                                self.solvent_sites[j][1],
                                                self.solvent_sites[i][2],
                                                self.solvent_sites[j][2])
                    q1 = self.solvent_sites[i][3]
                    q2 = self.solvent_sites[j][3]
                    Ur[:, i, j] = self.compute_UR_LJ(eps, sig, lam) + self.compute_UR_CMB(q1, q2, lam)
        return Ur

    def build_Ng_Pot(self, damping, lam):
        """
        Creates a matrix for the longe-range potential across the grid

        Returns
        -------
        Ur: ndarray
        An array containing long-range potential across grid
        """
        vv = True
        if vv == True:
            Ng = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float)
            for i,j in np.ndindex(self.nsv, self.nsv):
                if i == j:
                    q = self.solvent_sites[i][3]
                    Ng[:, i, j] = self.compute_UR_LR(q, q, damping, 1.0, lam)
                else:
                    q1 = self.solvent_sites[i][3]
                    q2 = self.solvent_sites[j][3]
                    Ng[:, i, j] = self.compute_UR_LR(q1, q2, damping, 1.0, lam)
        return Ng

    def build_rho(self):
        """
        Creates a matrix for the number density of a set of sites for molecules

        Returns
        -------
        rho_mat : ndarray
            An array with number densities of each site down the diagonal
        """
        return np.diag([prm[-1] for prm in self.solvent_sites])

    def RISM(self, wk, cr, vrlr, rho):
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
        I = np.identity(self.nsv)
        h = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float)
        trsr = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float)
        cksr = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float)
        crsr = cr + vrlr
        for i,j in np.ndindex(self.nsv, self.nsv):
            cksr[:, i, j] = self.grid.dht(crsr[:, i, j])
        for i in range(self.grid.npts):
            pre_inv = I - wk[i, :, :]@cksr[i, :, :]@rho
            h[i, :, :] = wk[i, :, :]@cksr[i, :, :]@(np.linalg.inv(pre_inv))@wk[i, :, :]
        for i,j in np.ndindex(self.nsv, self.nsv):
            trsr[:, i, j] = self.grid.idht(h[:, i, j] - cksr[:, i, j])
        return trsr

    def closure(self, vrsr, trsr):
        """
        Computes HNC CLosure equation in the form:

        c(r) = exp(-vr + tr) - tr - 1

        Parameters
        ----------
        vrsr: Short-range potential

        trsr: Indirect correlation function 3D-array

        Returns
        -------
        trsr: ndarray
        An array containing short-range indirection correlation function
        """
        return np.exp(-vrsr + trsr) - trsr - 1

    def dorism(self):
        """
        1. Initialises inputs
        2. Start charging process
        3. Iterate and cycle charging process till completion

        Parameters
        ----------
        Obtained from RismController Object

        Returns
        -------
        Iterated g(r), c(r), h(r)...
        """
        nlam = 10
        gr = np.zeros((self.grid.npts, self.nsv, self.nsv))
        wk = self.build_wk()
        rho = self.build_rho()
        itermax = 1000
        tol = 1E-7
        damp = 0.215
        for j in range(1, nlam+1):
            i = 0
            lam = 1.0 * j / nlam
            Ur = self.build_Ur(lam)
            Ng = self.build_Ng_Pot(1.0, lam)
            Ursr = (Ur - Ng)
            if j == 1:
                print("Building System...\n")
                cr = Ng
            else:
                print("Rebuilding System from previous cycle...\n")
                cr = cr_lam
            print("Iterating SSOZ Equations...\n")
            while i < itermax:
                if i % 100 == 0:
                    print("iteration: ", i)
                cr_prev = cr
                trsr = self.RISM(wk, cr, Ng, rho)
                cr_2 = self.closure(Ursr, trsr)
                cr_next = (1-damp)*cr_2 + damp*cr_prev
                cr = cr_next
                if np.sqrt(np.power((cr_next - cr_prev),2).sum() / (cr_next.shape[0] * np.power(cr_next.shape[2], 2))) < tol:
                    print("\nlambda: ", lam)
                    print("total iterations: ", i)
                    print("-------------------------")
                    break
                i+=1
            cr_lam = cr
        print("Iteration finished!\n")
        crt = cr - Ng
        trt = trsr + Ng
        gr1 = np.exp(-Ur + trt)
        print("First Max:\n", (np.amax(gr1[:, 0, 0])), self.grid.ri[np.argmax(gr1[:, 0, 0])])
        print("First Min:\n", (np.amin(gr1[np.argmax(gr1[:, 0, 0]):, 0, 0])), self.grid.ri[np.argwhere(gr1[:,0,0] == (np.amin(gr1[np.argmax(gr1[:, 0, 0]):, 0, 0])))].flatten()[0])
        plt.plot(self.grid.ri, gr1[:, 0, 0])
        plt.show()

if __name__ == "__main__":
    mol2 = RismController(1, 0, 1024, 20.48, Ar_fluid, [0])
    mol = RismController(3, 0, 2048, 10.24, water_sites, Solvent_Distances)
    mol2.dorism()
    #mol.dorism()