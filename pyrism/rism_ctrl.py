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
    uv_check: bool = field(init=False, default=False)
    vv: Core.RISM_Obj = field(init=False)
    uv: Core.RISM_Obj = field(init=False)
    pot: Potentials.Potential = field(init=False)
    solver: Solvers.Solver = field(init=False)
    closure: Closures.Closure = field(init=False)
    IE: IntegralEquations.IntegralEquation = field(init=False)
    IE_UV: IntegralEquations.IntegralEquation = field(init=False)
    solver_UV: Solvers.Solver = field(init=False)

    def initialise_controller(self):
        self.read_input()
        self.build_wk(self.vv)
        self.build_rho(self.vv)
        if self.uv_check:
            self.build_wk(self.uv)

    def do_rism(self):
        self.solve_vv(self.vv)
        self.write_vv(self.vv)
        if self.uv_check:
            self.solve_uv(self.vv, self.uv)
            self.write_uv(self.vv, self.uv)

    def read_input(self):
        inp = toml.load(self.fname)
        self.name = os.path.basename(self.fname).split(sep=".")[0]
        if "solvent" not in inp:
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

        if "solute" in inp:
            self.uv = Core.RISM_Obj(
                inp["system"]["temp"],
                inp["system"]["kT"],
                inp["system"]["charge_coeff"],
                inp["solute"]["nsu"],
                inp["solvent"]["nsv"],
                inp["solvent"]["nspv"],
                inp["solute"]["nspu"],
                inp["system"]["npts"],
                inp["system"]["radius"],
                inp["system"]["lam"],
            )
            self.uv_check = True
            solu_species = list(inp["solute"].items())[2 : self.uv.nsp1 + 2]
            for i in solu_species:
                self.add_species(i, self.uv)
                
        self.pot = Potentials.Potential(inp["params"]["potential"])
       
        self.closure = Closures.Closure(inp["params"]["closure"])
        self.IE = IntegralEquations.IntegralEquation(inp["params"]["IE"])
        if self.uv_check:
            self.IE_UV = IntegralEquations.IntegralEquation(inp["params"]["IE"] + "_UV")

        slv = Solvers.Solver(inp["params"]["solver"]).get_solver()
        self.solver = slv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"])
        if self.uv_check:
            slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
            self.solver_UV = slv_uv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], data_uv=self.uv)
        
    def add_species(self, spec_dat, data_object):
        new_spec = Core.Species(spec_dat[0])
        spdict = spec_dat[1]
        new_spec.set_density(spdict["dens"])
        new_spec.set_numsites(spdict["ns"])
        site_info = list(spdict.items())[2 : new_spec.ns + 2]
        for i in site_info:
            atom = Core.Site(i[0], i[1][0], np.asarray(i[1][1]))
            new_spec.add_site(atom)
            data_object.atoms.append(atom)
        data_object.species.append(new_spec)

    def distance_mat(self, dat):
        distance_arr = np.zeros((dat.ns1, dat.ns1), dtype=float)
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
        print(dat.w.shape)
        for i, j in np.ndindex(dat.ns1, dat.ns1):
            if dist_mat[i, j] < 0.0:
                dat.w[:, i, j] = zero_vec
            elif dist_mat[i, j] == 0.0:
                dat.w[:, i, j] = I
            else:
                dat.w[:, i, j] = np.sin(dat.grid.ki * dist_mat[i, j]) / (
                    dat.grid.ki * dist_mat[i, j]
                )


    def build_Ur(self, dat1, dat2, lam=1):
        sr_pot, mix = self.pot.get_potential()
        cou, _ = Potentials.Potential("cou").get_potential()
        for i, iat in enumerate(dat1.atoms):
            for j, jat in enumerate(dat2.atoms):
                i_sr_params = iat.params[:-1]
                j_sr_params = jat.params[:-1]
                qi = iat.params[-1]
                qj = jat.params[-1]
                if iat == jat:
                    dat1.u[:, i, j] = sr_pot(dat2.grid.ri, i_sr_params, lam) \
                        + cou(dat2.grid.ri, qi, qj, lam, dat2.amph)
                else:
                    mixed = mix(i_sr_params, j_sr_params)
                    dat1.u[:, i, j] = sr_pot(dat2.grid.ri, mixed, lam) \
                        + cou(dat2.grid.ri, qi, qj, lam, dat2.amph)

    def build_renorm(self, dat1, dat2, damping=1.0, lam=1):
        erfr, _ = Potentials.Potential("erfr").get_potential()
        erfk, _ = Potentials.Potential("erfk").get_potential()
        for i, iat in enumerate(dat1.atoms):
            for j, jat in enumerate(dat2.atoms):
                qi = iat.params[-1]
                qj = jat.params[-1]
                dat1.ur_lr[:, i, j] = erfr(dat2.grid.ri, qi, qj, damping, 1.0, lam, dat2.amph)
                dat1.uk_lr[:, i, j] = erfk(dat2.grid.ki, qi, qj, damping, lam, dat2.amph)

    def build_rho(self, dat):
        dens = []
        for isp in dat.species:
            for iat in isp.atom_sites:
                print(isp.dens)
                dens.append(isp.dens)
        dat.p = np.diag(dens)

    def write_csv(self, df, fname, ext, p, T):
        with open(fname+ext, 'w') as ofile:
            ofile.write("# density: {p}, temp: {T}\n".format(p=p[0][0], T=T))
            df.to_csv(ofile, index=False, header=True, mode='a')

    def write_vv(self, dat):
        all_sites = []
        for species in dat.species:
            for site in species.atom_sites:
                all_sites.append(site)
        gr = pd.DataFrame(dat.grid.ri, columns=["r"])
        cr = pd.DataFrame(dat.grid.ri, columns=["r"])
        tr = pd.DataFrame(dat.grid.ri, columns=["r"])
        for i, j in np.ndindex(dat.ns1, dat.ns2):
            lbl1 = all_sites[i].atom_type
            lbl2 = all_sites[j].atom_type
            gr[lbl1+"-"+lbl2] = dat.g[:, i, j]
            cr[lbl1+"-"+lbl2] = dat.c[:, i, j]
            tr[lbl1+"-"+lbl2] = dat.t[:, i, j]
        self.write_csv(gr, self.name, ".gvv", dat.p, dat.T)
        self.write_csv(cr, self.name, ".cvv", dat.p, dat.T)
        self.write_csv(tr, self.name, ".tvv", dat.p, dat.T)

    def write_uv(self, vv, uv):
        gr = pd.DataFrame(uv.grid.ri, columns=["r"])
        cr = pd.DataFrame(uv.grid.ri, columns=["r"])
        tr = pd.DataFrame(uv.grid.ri, columns=["r"])
        for i, iat in enumerate(uv.atoms):
            for j, jat in enumerate(vv.atoms):
                lbl1 = iat.atom_type
                lbl2 = jat.atom_type
                gr[lbl1+"-"+lbl2] = uv.g[:, i, j]
                cr[lbl1+"-"+lbl2] = uv.c[:, i, j]
                tr[lbl1+"-"+lbl2] = uv.t[:, i, j]
        self.write_csv(gr, self.name, ".guv", uv.p, uv.T)
        self.write_csv(cr, self.name, ".cuv", uv.p, uv.T)
        self.write_csv(tr, self.name, ".tuv", uv.p, uv.T)


    def solve_uv(self, dat1, dat2):
        clos = self.closure.get_closure()
        IE = self.IE_UV.get_IE()
        for j in range(1, dat2.nlam+1):
            lam = 1.0 * j / dat2.nlam
            self.build_Ur(dat2, dat1, lam)
            self.build_renorm(dat2, dat1, 1.0, lam)
            dat2.u_sr = dat2.u - dat2.ur_lr
            if j == 1:
                dat2.c = np.zeros_like(dat2.u_sr)
            else:
                pass
            self.solver_UV.solve_uv(IE, clos, lam)

        dat2.c -= dat2.B * dat2.ur_lr
        dat2.t += dat2.B * dat2.ur_lr
        dat2.g = 1.0 + dat2.c + dat2.t


    def solve_vv(self, dat):
        clos = self.closure.get_closure()
        IE = self.IE.get_IE()
        for j in range(1, dat.nlam+1):
            lam = 1.0 * j / dat.nlam
            self.build_Ur(dat, dat, lam)
            self.build_renorm(dat, dat, 1.0, lam)
            dat.u_sr = dat.u - dat.ur_lr
            if j == 1:
                dat.c = np.zeros_like(dat.u_sr)
            else:
                pass
            self.solver.solve(IE, clos, lam)

        dat.c -= dat.B * dat.ur_lr
        dat.t += dat.B * dat.ur_lr
        dat.g = 1.0 + dat.c + dat.t

if __name__ == "__main__":
    mol = RismController(sys.argv[1])
    mol.initialise_controller()
    mol.do_rism()
