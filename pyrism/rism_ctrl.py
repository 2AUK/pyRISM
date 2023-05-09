"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Initialises and solves the specified RISM problem.
"""
import os
os.environ['MKL_CBWR'] = 'AUTO'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'
#os.environ['MKL_VERBOSE'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toml
import sys
from pyrism import Closures
from pyrism import Core
from pyrism import IntegralEquations
from pyrism import Solvers
from pyrism import Potentials
from pyrism import Functionals
from pyrism import Util
from numba import njit, jit, prange
import time
import warnings

from dataclasses import dataclass, field

np.seterr(over="raise")
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

@dataclass
class RismController:
    """Initialises the parameters for the problem and starts the solver

    Attributes
    ----------
    fname : str
    Path to input file
    name : str
    Name of current RISM job (defaults to the file name)
    uv_check : bool
    Checks if a solute-solvent problem requires solving
    vv : Core.RISM_Obj
    Dataclass for parameters in solvent-solvent problem
    uv : Core.RISM_Obj
    Dataclass for parameters in solute-solvent problem
    solver : Solvers.Solver
    Dispatcher object to initialise solvent-solvent solver
    solver_UV : Solvers.Solver
    Dispatcher object to initialise solute-solvent solver
    closure : Closures.Closure
    Dispatcher object to initialise closures
    IE : IntegralEquations.IntegralEquation
    Dispatcher object to initialise solvent-solvent integral equation
    IE_UV : IntegralEquations.IntegralEquation
    Dispatcher object to initialise solute-solvent integral equation
    """
    fname: str
    name: str = field(init=False)
    uv_check: bool = field(init=False, default=False)
    vv: Core.RISM_Obj = field(init=False)
    uv: Core.RISM_Obj = field(init=False)
    pot: Potentials.Potential = field(init=False)
    solver: Solvers.Solver = field(init=False)
    solver_UV: Solvers.Solver = field(init=False)
    closure: Closures.Closure = field(init=False)
    IE: IntegralEquations.IntegralEquation = field(init=False)
    SFED: dict = field(init=False, default_factory=dict)
    SFE: dict = field(init=False, default_factory=dict)



    def initialise_controller(self):
        """ Reads input file `fname` to create `vv` and `uv` and
        builds the intra"""
        self.read_input()
        self.build_wk(self.vv)
        self.build_rho(self.vv)
         # Assuming infinite dilution, uv doesn't need p. Giving it vv's p makes later calculations easier
        if self.uv_check:
            self.uv.p = self.vv.p
            self.build_wk(self.uv)


    def do_rism(self, verbose=False):
        """ Solves the vv and uv (if applicable) problems and outputs the results"""
        if self.uv_check:
            self.solve(self.vv, dat2=self.uv, verbose=verbose)
        else:
            self.solve(self.vv, dat2=None, verbose=verbose)

    def read_input(self):
        """ Reads .toml input file, populates vv and uv dataclasses
        and properly initialises the appropriate potentials, solvers,
        closures, and integral equations"""
        inp = toml.load(self.fname)
        self.name = os.path.basename(self.fname).split(sep=".")[0]
        if "solvent" not in inp:
            raise ("No solvent data found!")
        else:
            self.vv = Core.RISM_Obj(
                inp["system"]["temp"],
                inp["system"]["kT"],
                inp["system"]["kU"],
                inp["system"]["charge_coeff"],
                inp["solvent"]["nsv"],
                inp["solvent"]["nsv"],
                inp["solvent"]["nspv"],
                inp["solvent"]["nspv"],
                inp["system"]["npts"],
                inp["system"]["radius"],
                inp["system"]["lam"],
            )
            solv_species = list(inp["solvent"].items())[2:self.vv.nsp1 + 2]
            for i in solv_species:
                self.add_species(i, self.vv)

        if "solute" in inp:
            self.uv = Core.RISM_Obj(
                inp["system"]["temp"],
                inp["system"]["kT"],
                inp["system"]["kU"],
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


        if inp["params"]["IE"] == "DRISM":
            IE = IntegralEquations.IntegralEquation(inp["params"]["IE"]).get_IE()
            if Util.total_moment(self.vv) is not None:
                Util.align_dipole(self.vv)
            else:
                print("No solvent dipole moment")
            if self.uv_check:
                if Util.total_moment(self.uv) is not None:
                    Util.align_dipole(self.uv)
                else:
                    print("No dipole moment, skipping alignment step...")
            if self.uv_check:
                self.IE = IE(self.vv, inp["params"]["diel"], inp["params"]["adbcor"], self.uv)
            else:
                self.IE = IE(self.vv, inp["params"]["diel"], inp["params"]["adbcor"])
        elif inp["params"]["IE"] == "XRISM-DB":
            IE = IntegralEquations.IntegralEquation(inp["params"]["IE"]).get_IE()
            if self.uv_check:
                self.IE = IE(inp["params"]["B"], self.vv, self.uv)
            else:
                self.IE = IE(inp["params"]["B"], self.vv)
            self.closure = Closures.Closure('r' + inp["params"]["closure"])
        else:
            IE = IntegralEquations.IntegralEquation(inp["params"]["IE"]).get_IE()
            if self.uv_check:
                self.IE = IE(self.vv, self.uv)
            else:
                self.IE = IE(self.vv)

        slv = Solvers.Solver(inp["params"]["solver"]).get_solver()
        slv = Solvers.Solver(inp["params"]["solver"]).get_solver()
        if inp["params"]["solver"] == "MDIIS":
            if 'mdiis_damping' in inp['params']:
                self.solver = slv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], m=inp["params"]["depth"], mdiis_damping=inp['params']['mdiis_damping'])
                if self.uv_check:
                    slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                    self.solver_UV = slv_uv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], data_uv=self.uv, m=inp["params"]["depth"], mdiis_damping=inp['params']['mdiis_damping'])

            else:
                self.solver = slv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], m=inp["params"]["depth"], mdiis_damping=inp['params']['picard_damping'])
                if self.uv_check:
                    slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                    self.solver_UV = slv_uv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], data_uv=self.uv, m=inp["params"]["depth"])
        elif inp["params"]["solver"] == "Gillan":
            self.solver = slv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], nbasis=inp["params"]["nbasis"])
            if self.uv_check:
                slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                self.solver_UV = slv_uv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], data_uv=self.uv, nbasis=inp["params"]["nbasis"])
        else:
            self.solver = slv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"])
            if self.uv_check:
                slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                self.solver_UV = slv_uv(self.vv, inp["params"]["tol"], inp["params"]["itermax"], inp["params"]["picard_damping"], data_uv=self.uv)


    def add_species(self, spec_dat, data_object):
        """Parses interaction sites and assigns them to relevant species

        Parameters
        ----------
        spec_dat: List
            Contains information on the current species pulled from the .toml file
        data_object: Core.RISM_Obj
            The dataclass to which species are being assigned
        """
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
        """Computes the Euclidean distance matrix while
        skipping distances between different sites of
        different species

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing information required for distance matrix

        Returns
        -------
        distance_arr: np.ndarray
            nD-Array with distances between atomic sites"""
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
        """Computes intramolecular correlation matrix in k-space for sites in same species

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing information required for intramolecular correlation

        Notes
        -----
        Calculated directly in k-space because not required in r-space:

        .. math:: \\omega(k) = \\frac{kl}{sin(kl)}
        """
        I = np.ones(dat.npts, dtype=np.float64)
        zero_vec = np.zeros(dat.npts, dtype=np.float64)
        dist_mat = self.distance_mat(dat)
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
        """Tabulates full short-range and Coulombic potential

        Parameters
        ----------
        dat1: Core.RISM_Obj
            Dataclass containing information required for potential
        dat2: Core.RISM_Obj
            Dataclass containing information required for potential
        lam: float
            :math: `\lambda` parameter for current charging cycle

        Notes
        -----
        For solvent-solvent problem `dat1` and `dat2` are the same,
        for the solute-solvent problem they both refer to the solvent
        and solute dataclasses respectively."""


        sr_pot, mix = self.pot.get_potential()
        cou, _ = Potentials.Potential("cou").get_potential()
        # dat1.u = build_Ur_impl(dat1.npts,
        #                        dat1.ns1,
        #                        dat1.ns2,
        #                        sr_pot,
        #                        mix,
        #                        cou,
        #                        dat1.atoms,
        #                        dat2.atoms,
        #                        dat2.grid.ri,
        #                        dat2.amph)
        for i, iat in enumerate(dat1.atoms):
            for j, jat in enumerate(dat2.atoms):
                i_sr_params = iat.params[:-1]
                j_sr_params = jat.params[:-1]
                qi = iat.params[-1]
                qj = jat.params[-1]
                if iat is jat:
                    dat1.u[:, i, j] = sr_pot(dat2.grid.ri, i_sr_params, lam) \
                        + cou(dat2.grid.ri, qi, qj, lam, dat2.amph)
                else:
                    mixed = mix(i_sr_params, j_sr_params)
                    dat1.u[:, i, j] = sr_pot(dat2.grid.ri, mixed, lam) \
                        + cou(dat2.grid.ri, qi, qj, lam, dat2.amph)

    def build_renorm(self, dat1, dat2, damping=1.0, lam=1):
        """Tabulates full short-range and Coulombic potential

        Parameters
        ----------
        dat1: Core.RISM_Obj
            Dataclass containing information required for potential
        dat2: Core.RISM_Obj
            Dataclass containing information required for potential
        damping: float
            damping parameter for adding a screened charge
        lam: float
            :math: `\lambda` parameter for current charging cycle

        Notes
        -----
        For solvent-solvent problem `dat1` and `dat2` are the same,
        for the solute-solvent problem they both refer to the solvent
        and solute dataclasses respectively."""
        erfr, _ = Potentials.Potential("erfr").get_potential()
        erfk, _ = Potentials.Potential("erfk").get_potential()
        for i, iat in enumerate(dat1.atoms):
            for j, jat in enumerate(dat2.atoms):
                qi = iat.params[-1]
                qj = jat.params[-1]
                dat1.ur_lr[:, i, j] = erfr(dat2.grid.ri, qi, qj, damping, 1.0, lam, dat2.amph)
                dat1.uk_lr[:, i, j] = erfk(dat2.grid.ki, qi, qj, damping, lam, dat2.amph)

    def build_rho(self, dat):
        """Builds diagonal matrix of species densities

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing species information
        """
        dens = []
        for isp in dat.species:
            for iat in isp.atom_sites:
                dens.append(isp.dens)
        dat.p = np.diag(dens)

    def write_csv(self, df, fname, ext, p, T, SFE=None):
        """Writes a dataframe to a .csv file with a header

        Parameters
        ----------
        df: pd.DataFrame
            Contains the functions to write to file
        fname: str
            Name of output file
        ext: str
            Extension of output file
        p: float
            Number density
        T: float
            Temperature"""
        with open(fname+ext, 'w') as ofile:
            if SFE is not None:
                ofile.write("# density: {p}, temp: {T}, HNC: {HNC}, GF: {GF}, KH: {KH}, PW: {PW}, HNCB: {RBC}\n".format(p=p[0][0], T=T, HNC=SFE['HNC'], GF=SFE['GF'], KH=SFE['KH'], PW=SFE['PW'], RBC=SFE['RBC']))
            else:
                ofile.write("# density: {p}, temp: {T}\n".format(p=p[0][0], T=T))
            df.to_csv(ofile, index=False, header=True, mode='a')

    def write_vv(self, dat):
        """Write solvent-solvent data to .csv file

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing correlation functions to output
        """
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
        """Write solute-solvent data to .csv file

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing correlation functions to output
        """
        gr = pd.DataFrame(uv.grid.ri, columns=["r"])
        cr = pd.DataFrame(uv.grid.ri, columns=["r"])
        tr = pd.DataFrame(uv.grid.ri, columns=["r"])
        dr = pd.DataFrame(uv.grid.ri, columns=["r"])
        d_GFr = pd.DataFrame(uv.grid.ri, columns=["r"])
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
    
    def write_output(self, duv_only=False):
        if duv_only and self.uv_check:
            self.SFED_write(self.uv.grid.ri, self.SFED, self.SFE, self.uv.p, self.uv.T)
        elif self.uv_check:
            self.write_vv(self.vv)
            self.write_uv(self.vv, self.uv)
            self.SFED_write(self.uv.grid.ri, self.SFED, self.SFE, self.uv.p, self.uv.T)
        else:
            self.write_vv(self.vv)

    def solve(self, dat1, dat2=None, verbose=False):
        """Start solving RISM problem

        Parameters
        ----------
        dat1: Core.RISM_Obj
            Dataclass containing all information for problem
        dat2: Core.RISM_Obj, optional
            Dataclass containing all information for problem

        Notes
        -----
        If `dat2` is not defined, the solvent-solvent problem is being solved.
        With `dat2`, the solute-solvent is solved.
        """
        fvv = np.exp(-dat1.B * dat1.u_sr) - 1.0
        if verbose == True:
            output_str="""\n-- pyRISM --\nRunning: {name},\nTemperature: {T},\nSolvent Density: {p}\nMethod: {IE}\nClosure: {clos}\nPotential: {pot},
            """.format(name=self.name,
                    T=str(dat1.T),
                    p=str(dat1.p[0][0]),
                    IE=self.IE.__class__.__name__,
                    clos=self.closure.get_closure().__name__,
                    pot=self.pot.get_potential()[0].__name__)
            print(output_str)
        if self.uv_check:
            fuv = np.exp(-dat2.B * dat2.u_sr) - 1.0
        for j in range(1, dat1.nlam+1):
            lam = 1.0 * j / dat1.nlam
            if j == 1:
                dat1.c = -dat1.B * dat1.ur_lr
                if self.uv_check:
                    dat2.c = -dat2.B * dat2.ur_lr
                else:
                    pass

            self.build_Ur(dat1, dat1, lam)
            self.build_renorm(dat1, dat1, 1.0, lam)
            dat1.u_sr = dat1.u - dat1.ur_lr
            self.solve_vv(lam, verbose)

            if self.uv_check:
                self.build_Ur(dat2, dat1, lam)
                self.build_renorm(dat2, dat1, 1.0, lam)
                dat2.u_sr = dat2.u - dat2.ur_lr
                self.solve_uv(lam, verbose)

        self.epilogue(dat1, dat2)

    def solve_uv(self, lam, verbose=False):
        """Call closure and integral equation functions and start solute-solvent solver

        Parameters
        ----------
        lam: float
            :math: `\\lambda` parameter for current charging cycle
        """
        clos = self.closure.get_closure()
        IE = self.IE.compute_uv
        self.solver_UV.solve_uv(IE, clos, lam, verbose)

    def solve_vv(self, lam, verbose=False):
        """Call closure and integral equation functions and start solvent-solvent solver

        Parameters
        ----------
        lam: float
            :math: `\\lambda` parameter for current charging cycle
        """
        clos = self.closure.get_closure()
        IE = self.IE.compute_vv
        self.solver.solve(IE, clos, lam, verbose)

    def integrate(self, SFE, dr):
        return dr * np.sum(SFE)

    def SFED_write(self, r, SFEDs, SFEs, p, T):
        dr = pd.DataFrame(r, columns=["r"])
        for SFED_key in SFEDs:
            dr[SFED_key] = SFEDs[SFED_key]
        self.write_csv(dr, self.name + "_SFED", ".duv", p, T, SFE=SFEs)


    def SFED_calc(self, dat2, vv=None):
        SFED_HNC = Functionals.Functional("HNC").get_functional()(dat2, vv)
        SFED_KH = Functionals.Functional("KH").get_functional()(dat2, vv)
        SFED_GF = Functionals.Functional("GF").get_functional()(dat2, vv)
        SFED_SC = Functionals.Functional("SC").get_functional()(dat2, vv)
        SFED_PW = Functionals.Functional("PW").get_functional()(dat2, vv)
        SFED_RBC = Functionals.Functional("RBC").get_functional()(dat2, vv)

        SFE_HNC = self.integrate(SFED_HNC, dat2.grid.d_r)
        SFE_KH = self.integrate(SFED_KH, dat2.grid.d_r)
        SFE_GF = self.integrate(SFED_GF, dat2.grid.d_r)
        SFE_SC = self.integrate(SFED_SC, dat2.grid.d_r)
        SFE_PW = self.integrate(SFED_PW, dat2.grid.d_r)
        SFE_RBC = self.integrate(SFED_RBC, dat2.grid.d_r)
        # SFE_text = "\n{clos_name}: {SFE_val} kcal/mol"

        # print(SFE_text.format(clos_name="KH", SFE_val=SFE_KH))
        # print(SFE_text.format(clos_name="HNC", SFE_val=SFE_HNC))
        # print(SFE_text.format(clos_name="GF", SFE_val=SFE_GF))

        self.SFED = {"HNC": SFED_HNC,
                     "KH": SFED_KH,
                     "GF": SFED_GF,
                     "SC": SFED_SC,
                     "PW": SFED_PW,
                     "RBC": SFED_HNC + SFED_RBC}
        self.SFE = {"HNC": SFE_HNC,
                    "KH": SFE_KH,
                    "GF": SFE_GF,
                    "SC": SFE_SC,
                    "PW": SFE_PW,
                    "RBC": SFE_HNC + SFE_RBC}


    def epilogue(self, dat1, dat2=None):
        """Computes final total, direct and pair correlation functions

        Parameters
         ----------
        dat1: Core.RISM_Obj
            Dataclass containing all information for final functions
        dat2: Core.RISM_Obj, optional
            Dataclass containing all information for final functions
        """

        dat1.c -= dat1.B * dat1.ur_lr
        dat1.t += dat1.B * dat1.ur_lr
        dat1.g = 1.0 + dat1.c + dat1.t
        dat1.h = dat1.t + dat1.c

        if self.uv_check:
            dat2.c -= dat2.B * dat2.ur_lr
            dat2.t += dat2.B * dat2.ur_lr
            dat2.g = 1.0 + dat2.c + dat2.t
            dat2.h = dat2.t + dat2.c
            self.SFED_calc(dat2, vv=dat1)

    def __isothermal_compressibility(self, dat):
        # WIP
        ck0 = 0.0
        for i in range(dat.grid.npts):
            ck0r = 0
            for j, k in np.ndindex(dat.ns1, dat.ns2):
                if j == k:
                    msym = 1
                else:
                    msym = 2
                ck0r += msym*np.diag(dat.p)[j]*np.diag(dat.p)[k]*dat.c[i, j, k]
            ck0 += ck0r * dat.grid.ri[i] ** 2
        ck0 *= 4.0 * np.pi * dat.grid.d_r
        return dat.B / (np.sum(dat.p) - ck0)

        """
        total_dens = np.sum(dat.p)
        ck = np.zeros_like(dat.c)

        # Don't think double counting is required here...
        cnt = np.full(dat.p.shape, 2)
        np.fill_diagonal(cnt, 1)

        int_cr = self.integrate(4.0 * np.pi * np.power(dat.grid.ri, 2)[:, np.newaxis, np.newaxis] * dat.p[np.newaxis, ...] @ dat.p[np.newaxis, ...] @ dat.c , self.uv.grid.d_r)
        #int_cr = self.integrate(np.power(total_dens, 2) * 4.0 * np.pi * np.power(dat.grid.ri, 2)[:, np.newaxis, np.newaxis] * dat.c , self.uv.grid.d_r)

        pc_0 = int_cr

        return dat.B / (total_dens - pc_0)
        """

    def __partial_molar_volume(self):
        # WIP
        # Taken from:
        # https://doi.org/10.1021/jp9608786

        int_cr = self.integrate(4.0 * np.pi * np.power(self.uv.grid.ri, 2)[:, np.newaxis, np.newaxis] * self.uv.c @ self.uv.p[np.newaxis, ...], self.uv.grid.d_r)
        #int_cr = self.integrate(np.power(np.sum(self.vv.p), 2) * 4.0 * np.pi * np.power(self.uv.grid.ri, 2)[:, np.newaxis, np.newaxis] * self.uv.c, self.uv.grid.d_r)

        X_t = self.isothermal_compressibility(self.vv)

        print(X_t.shape)

        return self.vv.kT * self.vv.T * X_t * (1 - int_cr)
        

    def __virial_pressure(self):
        # WIP
        duvv = np.zeros_like(self.vv.u)

        for i, j in np.ndindex(self.vv.ns1, self.vv.ns2):
            grad = np.gradient(self.vv.u[:, i, j], self.vv.grid.d_r)
            duvv[:, i, j] = grad

        pressure = (self.vv.g @ duvv) * np.power(self.uv.grid.ri, 3)[:, np.newaxis, np.newaxis]

        return (self.vv.kT * self.vv.T * np.sum(self.vv.p)) - self.vv.grid.d_r * 2.0 / 3.0 * np.pi * np.power(np.sum(self.vv.p), 2) * np.sum(pressure)

@jit
def build_Ur_impl(npts, ns1, ns2, sr_pot, mix, cou, atoms1, atoms2, r, charge_coeff, lam=1):
    """Tabulates full short-range and Coulombic potential

        Parameters
        ----------
        dat1: Core.RISM_Obj
            Dataclass containing information required for potential
        dat2: Core.RISM_Obj
            Dataclass containing information required for potential
        lam: float
            :math: `\lambda` parameter for current charging cycle

        Notes
        -----
        For solvent-solvent problem `dat1` and `dat2` are the same,
        for the solute-solvent problem they both refer to the solvent
        and solute dataclasses respectively."""
    u = np.zeros((npts, ns1, ns2), dtype=np.float64)
    for i, iat in enumerate(atoms1):
        for j, jat in enumerate(atoms2):
            i_sr_params = iat.params[:-1]
            j_sr_params = jat.params[:-1]
            qi = iat.params[-1]
            qj = jat.params[-1]
            if iat == jat:
                u[:, i, j] = sr_pot(r, i_sr_params, lam) \
                    + cou(r, qi, qj, lam, charge_coeff)
            else:
                mixed = mix(i_sr_params, j_sr_params)
                u[:, i, j] = sr_pot(r, mixed, lam) \
                    + cou(r, qi, qj, lam, charge_coeff)
        return u

if __name__ == "__main__":
    mol = RismController(sys.argv[1])
    mol.initialise_controller()
    if len(sys.argv) > 3:
        mol.vv.T = float(sys.argv[3])
        mol.vv.calculate_beta()
        if mol.uv_check:
            mol.uv.T = float(sys.argv[3])
            mol.uv.calculate_beta()
    mol.do_rism(verbose=True)
    if len(sys.argv) > 2:
        if bool(sys.argv[2]) == True:
            mol.write_output(duv_only=True)
