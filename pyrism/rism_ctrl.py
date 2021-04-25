"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Handles the primary functions
"""
import numpy as np
import mpmath as mp
import pandas as pd
from scipy.fft import dst, idst
from scipy.special import erf
from scipy.signal import argrelextrema
from scipy.spatial import distance_matrix
from scipy.optimize import root, minimize, newton_krylov, anderson
import matplotlib.pyplot as plt
import Data
import grid
import toml
import os
import sys

np.seterr(over="raise")


class RismController:
    def __init__(self, fname):
        self.fname = fname
        self.name = None
        self.nsu = None
        self.nsv = None
        self.npts = None
        self.radius = None
        self.charge_coeff = None
        self.solvent_sites = []
        self.dists = []
        self.itermax = None
        self.damp = None
        self.tol = None
        self.lam = None
        self.T = None
        self.kT = None
        self.cr = None
        self.tr = None
        self.gr = None
        self.Ur = None
        self.Ng = None
        self.Ursr = None
        self.wk = None
        self.rho = None
        self.clos = None
        self.read_input()
        self.beta = 1 / self.kT / self.T

    def read_input(self):
        inp = toml.load(self.fname)
        self.name = os.path.basename(self.fname).split(sep=".")[0]
        self.nsu = inp["solute"]["nsu"]
        self.nsv = inp["solvent"]["nsv"]
        self.npts = inp["system"]["npts"]
        self.radius = inp["system"]["radius"]
        solv_info = list(inp["solvent"].items())[1 : self.nsv + 1]
        coords = []
        for i in solv_info:
            i[1][0].insert(0, i[0])
            self.solvent_sites.append(i[1][0])
            coords.append(i[1][1])
        self.dists = distance_matrix(coords, coords)
        self.T = inp["system"]["temp"]
        self.kT = inp["system"]["kT"]
        self.charge_coeff = inp["system"]["charge_coeff"]
        self.itermax = inp["system"]["itermax"]
        self.lam = inp["system"]["lam"]
        self.damp = inp["system"]["picard_damping"]
        self.tol = inp["system"]["tol"]
        self.clos = inp["system"]["closure"]

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
                wk[:, i, j] = np.sin(self.grid.ki * self.dists[i, j]) / (
                    self.grid.ki * self.dists[i, j]
                )
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
            for i, j in np.ndindex(self.nsv, self.nsv):
                if i == j:
                    eps = self.solvent_sites[i][1]
                    sig = self.solvent_sites[i][2]
                    q = self.solvent_sites[i][3]
                    Ur[:, i, j] = self.compute_UR_LJ(
                        eps, sig, lam
                    ) + self.compute_UR_CMB(q, q, lam)
                else:
                    eps, sig = self.mixing_rules(
                        self.solvent_sites[i][1],
                        self.solvent_sites[j][1],
                        self.solvent_sites[i][2],
                        self.solvent_sites[j][2],
                    )
                    q1 = self.solvent_sites[i][3]
                    q2 = self.solvent_sites[j][3]
                    Ur[:, i, j] = self.compute_UR_LJ(
                        eps, sig, lam
                    ) + self.compute_UR_CMB(q1, q2, lam)
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
            for i, j in np.ndindex(self.nsv, self.nsv):
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
            for i, j in np.ndindex(self.nsv, self.nsv):
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

    def picard_step(self, cr_cur, cr_prev, damp):
        return cr_prev + damp * (cr_cur - cr_prev)

    def find_peaks(self):
        for i, j in np.ndindex(self.nsv, self.nsv):
            fmax = argrelextrema(self.gr[:, i, j], np.greater)
            fmin = argrelextrema(self.gr[:, i, j], np.less)
            lbl1 = self.solvent_sites[i][0]
            lbl2 = self.solvent_sites[j][0]
            print(lbl1 + "-" + lbl2)
            print("Maxima:")
            print("r", self.grid.ri[fmax])
            print("g(r)", self.gr[fmax, i, j].flatten())
            print("Minima:")
            print("r", self.grid.ri[fmin])
            print("g(r)", self.gr[fmin, i, j].flatten())
            print("\n")

    def plot_gr(self, save=False):
        for i in range(self.nsv):
            for j in range(i, self.nsv):
                lbl1 = self.solvent_sites[i][0]
                lbl2 = self.solvent_sites[j][0]
                plt.plot(self.grid.ri, self.gr[:, i, j], label=lbl1 + "-" + lbl2)
        plt.axhline(1, color="grey", linestyle="--", linewidth=2)
        plt.title("RDF of " + self.name + " at " + str(self.T) + " K")
        plt.xlabel("r/A")
        plt.ylabel("g(r)")
        plt.legend()
        if save == True:
            plt.savefig(self.name + "_RDF.eps", format="eps")
        plt.show()

    def write_data(self):

        gr = pd.DataFrame(self.grid.ri, columns=["r"])
        cr = pd.DataFrame(self.grid.ri, columns=["r"])
        tr = pd.DataFrame(self.grid.ri, columns=["r"])
        for i, j in np.ndindex(self.nsv, self.nsv):
            lbl1 = self.solvent_sites[i][0]
            lbl2 = self.solvent_sites[j][0]
            gr[lbl1 + "-" + lbl2] = self.gr[:, i, j]
            cr[lbl1 + "-" + lbl2] = self.cr[:, i, j]
            tr[lbl1 + "-" + lbl2] = self.tr[:, i, j]
        cr.to_csv(self.name + "_" + str(self.T) + "K.cvv", index=False)
        gr.to_csv(self.name + "_" + str(self.T) + "K.gvv", index=False)
        tr.to_csv(self.name + "_" + str(self.T) + "K.tvv", index=False)

    def cost(self, cr):
        cr_old = cr.reshape((self.grid.npts, self.nsv, self.nsv))
        trsr = self.RISM(self.wk, cr_old, self.Uklr, self.rho)
        cr_new = self.closure(self.Ursr, trsr, self.clos)
        self.cr = cr_new
        self.tr = trsr
        return (cr_new - cr_old).reshape(-1)

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
        Converged g(r), c(r), t(r)
        """
        nlam = self.lam
        self.wk = self.build_wk()
        self.rho = self.build_rho()
        itermax = self.itermax
        tol = self.tol
        damp = self.damp
        print(self.name)
        print("\n")
        print("System parameters\n")
        print("Temperature: ", str(self.T) + " K")
        print("-------------------------\n")
        for j in range(1, nlam + 1):
            vecfr = []
            vecgr = []
            A = np.zeros((2, 2), dtype=float)
            b = np.zeros(2, dtype=float)
            i = 0
            lam = 1.0 * j / nlam
            Ur = self.build_Ur(lam)
            Ng = self.build_Ng_Pot(1.0, lam)
            self.Ursr = Ur - Ng
            # Ursr[Ursr < urmin] = urmin
            self.Uklr = self.build_Ng_Pot_k(1.0, lam)
            fr = np.exp(-1 * (self.Ursr)) - 1.0
            if j == 1:
                print("Building System...\n")
                cr = fr
            else:
                print("Rebuilding System from previous cycle...\n")
                cr = cr_lam

            print(lam)
            print("Iterating SSOZ Equations...\n")
            # min_result = anderson(self.cost, cr.reshape(-1), verbose=True, M=20)
            while i < itermax:
                cr_prev = cr
                trsr = self.RISM(self.wk, cr, self.Uklr, self.rho)
                cr_A = self.closure(self.Ursr, trsr, self.clos)
                if i < 3:
                    vecfr.append(cr_prev)
                    cr_next = self.picard_step(cr_A, cr_prev, damp)
                    vecgr.append(cr_A)
                else:
                    vecdr = np.asarray(vecgr) - np.asarray(vecfr)
                    dn = vecdr[-1].flatten()
                    d01 = (vecdr[-1] - vecdr[-2]).flatten()
                    d02 = (vecdr[-1] - vecdr[-3]).flatten()
                    A[0, 0] = np.inner(d01, d01)
                    A[0, 1] = np.inner(d01, d02)
                    A[1, 0] = np.inner(d01, d02)
                    A[1, 1] = np.inner(d02, d02)
                    b[0] = np.inner(dn, d01)
                    b[1] = np.inner(dn, d02)
                    c = np.linalg.solve(A, b)
                    cr_next = (
                        (1 - c[0] - c[1]) * vecgr[-1]
                        + c[0] * vecgr[-2]
                        + c[1] * vecgr[-3]
                    )
                    vecfr.append(cr_prev)
                    vecgr.append(cr_A)
                    vecgr.pop(0)
                    vecfr.pop(0)
                y = np.abs(cr_next - cr_prev)
                rms = np.sqrt(
                    self.grid.d_r
                    * np.power((cr_next - cr_prev), 2).sum()
                    / (np.prod(cr_next.shape))
                )
                if i % 100 == 0:
                    print("iteration: ", i, "\tRMS: ", rms, "\tDiff: ", np.amax(y))
                if rms < tol:
                    print("\nlambda: ", lam)
                    print("total iterations: ", i)
                    print("RMS: ", rms)
                    print("Diff: ", np.amax(y))
                    print("-------------------------")
                    break
                i += 1
                if i == itermax:
                    print("\nlambda: ", lam)
                    print("total iterations: ", i)
                    print("RMS: ", rms)
                    print("Diff: ", np.amax(y))
                    print("-------------------------")
                cr = cr_next
            # print(min_result)
            cr_lam = cr
        print("Iteration finished!\n")
        self.cr = cr - Ng
        self.tr = trsr + Ng
        self.gr = 1 + self.cr + self.tr
        self.Ur = Ur
        self.Ng = Ng
        self.find_peaks()
        self.plot_gr()
        # self.write_data()


if __name__ == "__main__":
    mol = RismController(sys.argv[1])
    mol.dorism()