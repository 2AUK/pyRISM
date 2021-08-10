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
import Closures
import Core
import IntegralEquations
import Solvers
import Potentials

from dataclasses import dataclass, field

np.seterr(over="raise")


@dataclass
class RismController:

    # Input filename
    fname: str
    name: str = field(init=False)
    vv: Core.RISM_Obj = field(init=False)
    uv: Core.RISM_Obj = field(init=False)
    pot: Potentials.Potential = field(init=False)
    solver: Solvers.Solver = field(init=False)
    closure: Closures.Closure = field(init=False)
    IE: IntegralEquations.IntegralEquation = field(init=False)

    def initialise_controller(self):
        print("init")
        self.read_input()
        self.build_wk(self.vv)
        self.build_rho(self.vv)
        self.solve_system(self.vv)

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
                
        self.pot = Potentials.Potential(inp["params"]["potential"])
       
        self.closure = Closures.Closure(inp["params"]["closure"])
        self.IE = IntegralEquations.IntegralEquation(inp["params"]["IE"])

        slv = Solvers.Solver(inp["params"]["solver"]).get_solver()
        self.solver = slv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"])
        
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
        I = np.ones(dat.npts, dtype=np.float64)
        zero_vec = np.zeros(dat.npts, dtype=np.float64)
        dist_mat = self.distance_mat(dat)
        for i, j in np.ndindex(dat.ns1, dat.ns2):
            if dist_mat[i, j] < 0.0:
                dat.w[:, i, j] = zero_vec
            elif dist_mat[i, j] == 0.0:
                dat.w[:, i, j] = I
            else:
                dat.w[:, i, j] = np.sin(dat.grid.ki * dist_mat[i, j]) / (
                    dat.grid.ki * dist_mat[i, j]
                )


    def build_Ur(self, dat, lam=1):
        sr_pot, mix = self.pot.get_potential()
        cou, _ = Potentials.Potential("cou").get_potential()
        i = 0
        for isp in dat.species:
            for iat in isp.atom_sites:
                j = 0
                for jsp in dat.species:
                    for jat in jsp.atom_sites:
                        i_sr_params = iat.params[:-1]
                        j_sr_params = jat.params[:-1]
                        qi = iat.params[-1]
                        qj = jat.params[-1]
                        if iat == jat:
                            dat.u[:, i, j] = sr_pot(dat.grid.ri, i_sr_params, lam, dat.B) \
                                + cou(dat.grid.ri, qi, qj, lam, dat.B, dat.amph)
                        else:
                            mixed = mix(iat.params, jat.params)
                            dat.u[:, i, j] = sr_pot(dat.grid.ri, mixed, lam, dat.B) \
                                + cou(dat.grid.ri, qi, qj, lam, dat.B, dat.amph)
                        j += 1
                i += 1

    def build_renorm(self, dat, damping = 1.0, lam = 1):
        erfr, _ = Potentials.Potential("erfr").get_potential()
        erfk, _ = Potentials.Potential("erfk").get_potential()
        i = 0
        for isp in dat.species:
            for iat in isp.atom_sites:
                j = 0
                for jsp in dat.species:
                    for jat in jsp.atom_sites:
                        qi = iat.params[-1]
                        qj = jat.params[-1]
                        dat.ur_lr[:, i, j] = erfr(dat.grid.ri, qi, qj, damping, 1.0, lam, dat.B, dat.amph)
                        dat.uk_lr[:, i, j] = erfk(dat.grid.ki, qi, qj, damping, lam, dat.B, dat.amph)
                        j += 1
                i += 1

    def build_rho(self, dat):
        dens = []
        for isp in dat.species:
            for iat in isp.atom_sites:
                print(isp.dens)
                dens.append(isp.dens)
        dat.p = np.diag(dens)

    def solve_system(self, dat):
        clos = self.closure.get_closure()
        IE = self.IE.get_IE()
        for j in range(1, dat.nlam+1):
            lam = 1.0 * j / dat.nlam
            self.build_Ur(dat, lam)
            self.build_renorm(dat, 1.0, lam)
            dat.u_sr = dat.u - dat.ur_lr
            if j == 1:
                dat.c = np.exp(-1 * (dat.u_sr)) - 1.0
            else:
                pass
            self.solver.solve(IE, clos, lam)
                        
        dat.c -= dat.B * dat.ur_lr
        dat.t += dat.B * dat.ur_lr
        gr = 1 + dat.c + dat.t
        plt.plot(dat.grid.ri, gr[:, 0, 0])
        plt.show()

if __name__ == "__main__":
    mol = RismController(sys.argv[1])
    mol.initialise_controller()
