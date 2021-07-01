"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Handles the primary functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toml
import os
import sys
import attr
import Closures
import Core
import IntegralEquations
import Solvers
import Potentials

np.seterr(over="raise")


@attr.s
class RismController:

    # Input filename
    fname: str = attr.ib()
    name: str = attr.ib(init=False)
    vv: Core.RISM_Obj = attr.ib(init=False)
    uv: Core.RISM_Obj = attr.ib(init=False)

    def initialise_controller(self):
        print("init")
        self.read_input()

    def read_input(self):
        inp = toml.load(self.fname)
        self.name = os.path.basename(self.fname).split(sep=".")[0]
        if not inp["solvent"]:
            raise ("No solvent data found!")
        else:
            self.vv = Core.RISM_Obj(
                inp["system"]["temp"],
                inp["system"]["kT"],
                inp["system"]["charge_coeff"],
                inp["solvent"]["nsv"],
                inp["solvent"]["nsv"],
                inp["solvent"]["nspv"],
                inp["solvent"]["nspv"],
                inp["system"]["npts"],
                inp["system"]["radius"],
                inp["system"]["lam"],
            )
            solv_species = list(inp["solvent"].items())[2 : self.vv.nsp1 + 2]
            for i in solv_species:
                self.add_species(i, self.vv)

        if inp["solute"]:
            self.uv = Core.RISM_Obj(
                inp["system"]["temp"],
                inp["system"]["kT"],
                inp["system"]["charge_coeff"],
                inp["solute"]["nsu"],
                inp["solvent"]["nsv"],
                inp["solute"]["nspu"],
                inp["solvent"]["nspv"],
                inp["system"]["npts"],
                inp["system"]["radius"],
                inp["system"]["lam"],
            )
            solu_species = list(inp["solute"].items())[2 : self.uv.nsp1 + 2]
            for i in solu_species:
                self.add_species(i, self.uv)

        self.build_wk(self.vv)
        # print(self.vv)

    def add_species(self, spec_dat, data_object):
        new_spec = Core.Species(spec_dat[0])
        spdict = spec_dat[1]
        new_spec.set_density(spdict["dens"])
        new_spec.set_numsites(spdict["ns"])
        site_info = list(spdict.items())[2 : new_spec.ns + 2]
        for i in site_info:
            new_spec.add_site(Core.Site(i[0], i[1][0], np.asarray(i[1][1])))
        data_object.species.append(new_spec)

    def distance_mat(self, dat):
        distance_arr = np.zeros((dat.ns1, dat.ns2), dtype=float)
        i = 0
        for isp in dat.species:
            for iat in isp.atom_sites:
                j = 0
                for jsp in dat.species:
                    for jat in jsp.atom_sites:
                        if isp != jsp:
                            distance_arr[i, j] = -1
                        else:
                            distance_arr[i, j] = np.linalg.norm(iat.coords - jat.coords)
                        j += 1
                i += 1
        return distance_arr

    def build_wk(self, dat):
        wk = np.zeros((dat.npts, dat.ns1, dat.ns2), dtype=np.float64)
        I = np.ones(dat.npts, dtype=np.float64)
        zero_vec = np.zeros(dat.npts, dtype=np.float64)
        dist_mat = self.distance_mat(dat)
        for i, j in np.ndindex(dat.ns1, dat.ns2):
            if dist_mat[i, j] < 0.0:
                wk[:, i, j] = zero_vec
            elif dist_mat[i, j] == 0.0:
                wk[:, i, j] = I
            else:
                wk[:, i, j] = np.sin(dat.grid.ki * dist_mat[i, j]) / (
                    dat.grid.ki * dist_mat[i, j]
                )
        return wk


"""
    def build_Ur(self, lam):

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
 """

if __name__ == "__main__":
    mol = RismController(sys.argv[1])
    mol.initialise_controller()
