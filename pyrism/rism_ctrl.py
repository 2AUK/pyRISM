"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Handles the primary functions
"""
import numpy as np
import mpmath as mp
from scipy.fft import dst, idst
from scipy.special import erf, expit
from scipy.signal import argrelextrema
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import grid

np.seterr(over='raise')

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
Temp = 293.16
kTkcal = 0.00198720414667
kT = 1
beta = 1 / (kT * Temp)
KE2PK = 167101.0
KE2NAC = 332.064
charge_coeff = KE2PK
# Site = [Site1, Site2, ... , SiteN]
# SiteN = [Atomic symbol, eps, sig, charge, rho]
#Argon fluid. Lennard-Jones parameters from Rahman (already divided by kB)
#Number density computed from equation $N = \frac{\N_A}{M}\rho$ where \rho is the mass density (1.394 g/cm^3)
Ar_fluid = [["Ar", 120, 3.4, 0, 0.021017479720736955]]

Ar_dist = np.asarray([0.00])

#cSPCE model from amber
water_sites = [["O", 78.15, 3.16572, -0.8476, 0.033314],
                 ["H", 7.815, 1.16572, 0.4238, 0.033314],
                 ["H", 7.815, 1.16572, 0.4238, 0.033314]]

Solvent_Distances = np.asarray([[0.0, 1.0, 1.0],
                                [1.0, 0.0, 1.633],
                                [1.0, 1.633, 0.0]])

TIP3P_sites = [["O", 78.15, 3.16572, -0.8476, 0.033314],
                 ["H", 7.815, 1.16572, 0.4238, 0.033314],
                 ["H", 7.815, 1.16572, 0.4238, 0.033314]]

TIP3P_Distances = np.asarray([[0.0, 1.0, 1.0],
                                [1.0, 0.0, 1.633],
                                [1.0, 1.633, 0.0]])

HR1981 = [["N", 44, 3.341, 0.2, 0.01867],
            ["N", 44, 3.341, -0.2, 0.01867]]

HR1981_dist = np.asarray([[0.0, 1.1],
                          [1.1, 0.0]])

HR1981_NN = [["N", 44, 3.341, 0.0, 0.01867],
            ["N", 44, 3.341, 0.0, 0.01867]]

HR1982_HCL_II = [["H", 20, 2.735, 0.2, 0.018],
                ["Cl", 259, 3.353, -0.2, 0.018]]

HR1982_HCL_II_dist = np.asarray([[0.0, 1.257],
                            [1.257, 0.0]])

HR1982_HCL_III = [["H", 20, 0.4, 0.2, 0.018],
                ["Cl", 259, 3.353, -0.2, 0.018]]

HR1982_HCL_III_dist = np.asarray([[0.0, 1.3],
                            [1.3, 0.0]])

HR1982_BR2_I = [["Br", 245.7, 3.63, -0.48, 0.01175728342],
                ["Br", 245.7, 3.63, -0.48, 0.01175728342],
                ["X", 0.0724, 1.0, 0.96, 0.01175728342]]

HR1982_BR2_III = [["Br", 130, 3.63, -0.3, 0.01175728342],
                ["Br", 130, 3.63, -0.3, 0.01175728342],
                ["X", 0.0724, 1.0, 0.6, 0.01175728342]]
            
HR1982_BR2_IV = [["Br", 130, 3.63, 0.0, 0.01175728342],
                ["Br", 130, 3.63, 0.0, 0.01175728342],
                ["X", 0.0724, 1.0, 0.0, 0.01175728342]]

HR1982_BR2_I_dist = np.asarray([[0.0, 2.284, 1.142],
                                [2.284, 0.0, 1.142],
                                [1.142, 1.142, 0.0]])

br2_coord =[[0.0, 0.0, 0.0],
            [0.0, 0.0, 2.284],
            [0.0, 0.0, 1.142]]

dist_mat = distance_matrix(br2_coord, br2_coord)
print(dist_mat)
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

    def compute_UR_LJ_C(self, C6, C12, lam) -> np.ndarray:

        return beta * ( (C12 / self.grid.ri**12) - (C6 / self.grid.ri**6) )

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
        return  lam * beta * charge_coeff * q1 * q2 / self.grid.ri

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

    def compute_UK_LR(self, q1, q2, damping, lam):
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
        return lam * beta * 4 * np.pi * q1 * q2 * charge_coeff * np.exp( -1.0 * self.grid.ki**2 / (4.0 * damping**2)) / self.grid.ki**2

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
        wk = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
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
            Ur = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
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
            Ng = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
            for i,j in np.ndindex(self.nsv, self.nsv):
                if i == j:
                    q = self.solvent_sites[i][3]
                    Ng[:, i, j] = self.compute_UR_LR(q, q, damping, 1.0, lam)
                else:
                    q1 = self.solvent_sites[i][3]
                    q2 = self.solvent_sites[j][3]
                    Ng[:, i, j] = self.compute_UR_LR(q1, q2, damping, 1.0, lam)
        return Ng


    def build_Ng_Pot_k(self, damping, lam):
        """
        Creates a matrix for the longe-range potential across the grid

        Returns
        -------
        Ur: ndarray
        An array containing long-range potential across grid
        """
        vv = True
        if vv == True:
            Ng_k = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
            for i,j in np.ndindex(self.nsv, self.nsv):
                if i == j:
                    q = self.solvent_sites[i][3]
                    Ng_k[:, i, j] = self.compute_UK_LR(q, q, damping, lam)
                else:
                    q1 = self.solvent_sites[i][3]
                    q2 = self.solvent_sites[j][3]
                    Ng_k[:, i, j] = self.compute_UK_LR(q1, q2, damping, lam)
        return Ng_k

    def build_rho(self):
        """
        Creates a matrix for the number density of a set of sites for molecules

        Returns
        -------
        rho_mat : ndarray
            An array with number densities of each site down the diagonal
        """
        return np.diag([prm[-1] for prm in self.solvent_sites])

    def RISM(self, wk, cr, vklr, rho):
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
        h = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
        trsr = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
        cksr = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
        #vklr = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
        for i,j in np.ndindex(self.nsv, self.nsv):
            cksr[:, i, j] = self.grid.dht(cr[:, i, j])
            #vklr[:, i, j] = self.grid.dht(vrlr[:, i, j]) 
            cksr[:, i, j] -= vklr[:, i, j]
        for i in range(self.grid.npts):
            #wc = np.matmul(wk[i, :, :], cksr[i, :, :])
            #pwc = np.matmul(wc, rho)
            #ipwc = np.linalg.inv(I - pwc)
            #h[i, :, :] = np.matmul(np.matmul(wc, ipwc), wk[i, :, :])
            iwcp = np.linalg.inv(I - wk[i, :, :]@cksr[i, :, :]@rho)
            wcw = wk[i, :, :]@cksr[i, :, :]@wk[i, :, :]
            h[i, :, :] = iwcp@wcw - cksr[i, :, :]
        for i,j in np.ndindex(self.nsv, self.nsv):
            trsr[:, i, j] = self.grid.idht(h[:, i, j] - vklr[:, i, j])
        return trsr

    def closure(self, vrsr, trsr):
        """
        Computes HNC Closure equation in the form:

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
        #print(np.exp(trsr))
        #exp_arr = np.frompyfunc(mp.exp, 1, 1)
        return np.exp(-vrsr + trsr) - 1.0 - trsr

    def picard_step(self, cr_cur, cr_prev, damp):
        return cr_prev + damp*(cr_cur - cr_prev)
        #return damp*cr_cur + (1-damp)*cr_prev


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
        gr = np.zeros((self.grid.npts, self.nsv, self.nsv), dtype=np.float64)
        wk = self.build_wk()
        rho = self.build_rho()
        itermax = 10000
        tol = 1E-7
        damp = 0.0001
        urmin = -30
        for j in range(1, nlam+1):
            vecfr = []
            vecgr = []
            A = np.zeros((2,2), dtype=float)
            b = np.zeros(2, dtype=float)
            i = 0
            lam = 1.0 * j / nlam
            Ur = self.build_Ur(lam)
            Ng = self.build_Ng_Pot(1.0, lam)
            Ursr = (Ur - Ng)
            #Ursr[Ursr < urmin] = urmin
            Uklr = self.build_Ng_Pot_k(1.0, lam)
            fr = np.exp(-Ursr) - 1.0
            if j == 1:
                print("Building System...\n")
                cr = fr
            else:
                print("Rebuilding System from previous cycle...\n")
                cr = cr_lam
            print("Iterating SSOZ Equations...\n")
            while i < itermax:
                cr_prev = cr
                trsr = self.RISM(wk, cr, Uklr, rho)
                cr_A = self.closure(Ursr, trsr)
                if i < 3:
                    vecfr.append(cr_prev)
                    cr_next = self.picard_step(cr_A, cr_prev, damp)
                    vecgr.append(cr_A)
                else:
                    vecdr = np.asarray(vecgr) - np.asarray(vecfr)
                    dn = vecdr[-1].flatten()
                    d01 = (vecdr[-1] - vecdr[-2]).flatten()
                    d02 = (vecdr[-1] - vecdr[-3]).flatten()
                    A[0,0] = np.inner(d01, d01)
                    A[0,1] = np.inner(d01, d02)
                    A[1,0] = np.inner(d01, d02)
                    A[1,1] = np.inner(d02, d02)
                    b[0] = np.inner(dn, d01)
                    b[1] = np.inner(dn, d02)
                    c = np.linalg.solve(A, b)
                    cr_next = (1 - c[0] - c[1])*vecgr[-1] + c[0]*vecgr[-2] + c[1]*vecgr[-3]
                    vecfr.append(cr_prev)
                    vecgr.append(cr_A)
                    vecgr.pop(0)
                    vecfr.pop(0)
                cr = cr_next
                y = np.abs(cr_next - cr_prev)
                rms = np.sqrt(self.grid.d_r * np.power((cr_next - cr_prev), 2).sum() / (np.prod(cr_next.shape)))
                if i % 100 == 0:
                    print("iteration: ", i, "\tRMS: ", rms, "\tDiff:, ", np.amax(y))
                if rms < tol and np.amax(y) < tol:
                    print("\nlambda: ", lam)
                    print("total iterations: ", i)
                    print("RMS: ", rms)
                    print("Diff: ", np.amax(y))
                    print("-------------------------")
                    break
                i+=1
                if i == itermax:
                    print("\nlambda: ", lam)
                    print("total iterations: ", i)
                    print("RMS: ", rms)
                    print("Diff: ", np.amax(y))
                    print("-------------------------")
            cr_lam = cr
        print("Iteration finished!\n")
        crt = cr - Ng
        trt = trsr + Ng
        #print("r:\n", self.grid.ri)
        #print("crt:\n", crt[:, 0, 0])
        #print("trt:\n", trt[:, 0, 0])
        #print("Ng:\n", Ng[:, 0, 0])
        #print("Ur:\n", Ur[:, 0, 0])
        #print("fr:\n", fr[:, 0, 0])
        gr1 = 1 + crt + trt
        fmax = argrelextrema(gr1[:,0,0], np.greater)
        fmin = argrelextrema(gr1[:,0,0], np.less)
        print("Maxima:")
        print(self.grid.ri[fmax])
        print(gr1[fmax, 0, 0])
        print("Minima:")
        print(self.grid.ri[fmin])
        print(gr1[fmin, 0, 0])
        #print("First Max:\n", (np.amax(gr1[:, 0, 0])), self.grid.ri[np.argmax(gr1[:, 0, 0])])
        #print("First Min:\n", (np.amin(gr1[np.argmax(gr1[:, 0, 0]):, 0, 0])), self.grid.ri[np.argwhere(gr1[:,0,0] == (np.amin(gr1[np.argmax(gr1[:, 0, 0]):, 0, 0])))].flatten()[0])
        #print("First Max:\n", (np.amax(gr1[:, 0, 1])), self.grid.ri[np.argmax(gr1[:, 0, 1])])
        #print("First Min:\n", (np.amin(gr1[np.argmax(gr1[:, 0, 1]):, 0, 1])), self.grid.ri[np.argwhere(gr1[:,0,1] == (np.amin(gr1[np.argmax(gr1[:, 0, 1]):, 0, 1])))].flatten()[0])
        #print("First Max:\n", (np.amax(gr1[:, 1, 1])), self.grid.ri[np.argmax(gr1[:, 1, 1])])
        #print("First Min:\n", (np.amin(gr1[np.argmax(gr1[:, 1, 1]):, 1, 1])), self.grid.ri[np.argwhere(gr1[:,1,1] == (np.amin(gr1[np.argmax(gr1[:, 1, 1]):, 1, 1])))].flatten()[0])        
        plt.plot(self.grid.ri, gr1[:, 0, 0], 'k-')
        plt.plot(self.grid.ri, gr1[:, 0, 1], 'r--')
        plt.plot(self.grid.ri, gr1[:, 1, 0], 'b--')
        plt.plot(self.grid.ri, gr1[:, 1, 1], 'g-.')
        plt.axhline(1, color='grey', linestyle="--", linewidth=2)
        #plt.title("RDF of Water at " + str(Temp) + "K")
        plt.xlabel("r(A)")
        plt.ylabel("g(r)")
        plt.legend()
        plt.savefig('ARRDF.eps', format='eps')
        plt.show()
        plt.axhline(0, color='grey', linestyle="--", linewidth=2)
        plt.xlim([0.0, self.grid.radius/2])
        plt.ylim([-3, 3])
        plt.plot(self.grid.ri, gr1[:, 0, 0] - 1, 'k-', label="Total")
        plt.plot(self.grid.ri, cr[:, 0, 0], 'k--', label="Direct")
        plt.plot(self.grid.ri, trsr[:, 0, 0], 'k-.', label="Indirect")
        plt.xlabel("r(A)")
        plt.ylabel("Correlation")
        plt.legend()
        plt.savefig('corrArbnw.eps', format='eps')
        plt.show()
        plt.axhline(0, color='grey', linestyle="--", linewidth=2)
        plt.xlim([0.0, self.grid.radius/2])
        plt.ylim([-3, 3])
        plt.plot(self.grid.ri, cr[:, 0, 0], 'k--', label="Direct")
        plt.xlabel("r(A)")
        plt.ylabel("Correlation")
        plt.plot(self.grid.ri, Ursr[:, 0, 0], 'k-', label="Pair Potential")
        plt.legend()
        plt.savefig('crurAr.eps', format='eps')
        plt.show()


if __name__ == "__main__":
    mol2 = RismController(1, 0, 2048, 20.48, Ar_fluid, [0])
    mol = RismController(3, 0, 2048, 20.48, water_sites, Solvent_Distances)
    hr1981 = RismController(2, 0, 2048, 20.48, HR1981, HR1981_dist)
    hr1981nn = RismController(2, 0, 2048, 20.48, HR1981_NN, HR1981_dist)
    hr1982_hcl_ii = RismController(2, 0, 2048, 20.48, HR1982_HCL_II, HR1982_HCL_II_dist)
    hr1982_hcl_iii = RismController(2, 0, 2048, 20.48, HR1982_HCL_III, HR1982_HCL_III_dist)
    hr1982_br2_i = RismController(3, 0, 2048, 20.48, HR1982_BR2_I, HR1982_BR2_I_dist)
    hr1982_br2_iii = RismController(3, 0, 2048, 20.48, HR1982_BR2_III, HR1982_BR2_I_dist)
    hr1982_br2_iv = RismController(3, 0, 2048, 20.48, HR1982_BR2_IV, HR1982_BR2_I_dist)
    #mol2.dorism()
    #mol.dorism()
    #hr1981.dorism()
    #hr1981nn.dorism()
    #hr1982_hcl_ii.dorism()
    #hr1982_hcl_iii.dorism()
    #hr1982_br2_i.dorism() #Doesn't work without an arbitrary precision lib
    #hr1982_br2_iii.dorism()
    #hr1982_br2_iv.dorism()
