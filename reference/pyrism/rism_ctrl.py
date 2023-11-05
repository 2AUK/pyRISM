"""
rism_ctrl.py
A pedagogical implementation of the RISM equations

Initialises and solves the specified RISM problem.
"""
import os

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
from pyrism.librism import RISMDriver
from dataclasses import dataclass, field
from enum import Enum


np.seterr(over="raise")
np.set_printoptions(precision=20)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


@dataclass
class GillanSettings:
    nbasis: int


@dataclass
class MDIISSettings:
    depth: int
    damping: float


@dataclass
class SolverSettings:
    picard_damping: float
    max_iter: int
    tolerance: float
    gillan_settings: object = None
    mdiis_settings: object = None


@dataclass
class SolverConfig:
    solver: str
    settings: object


@dataclass
class DataConfig:
    temp: float
    kt: float
    ku: float
    amph: float
    drism_damping: float
    dielec: float
    nsv: int
    nsu: int
    nspv: int
    nspu: int
    npts: int
    radius: float
    nlambda: int
    preconverged: str
    solvent_atoms: list = field(default_factory=list)
    solute_atoms: list = field(default_factory=list)
    solvent_species: list = field(default_factory=list)
    solute_species: list = field(default_factory=list)


@dataclass
class OperatorConfig:
    integral_equation: str
    closure: str


@dataclass
class PotentialConfig:
    nonbonded: str
    coulombic: str
    renormalisation_real: str
    renormalisation_fourier: str


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
        """Reads input file `fname` to create `vv` and `uv` and
        builds the intra"""
        self.read_input()
        self.build_wk(self.vv)
        self.build_rho(self.vv)
        # Assuming infinite dilution, uv doesn't need p. Giving it vv's p makes later calculations easier
        if self.uv_check:
            self.uv.p = self.vv.p
            self.build_wk(self.uv)

    def do_rism(self, verbose=False):
        """Solves the vv and uv (if applicable) problems and outputs the results"""
        if self.uv_check:
            self.solve(self.vv, dat2=self.uv, verbose=verbose)
        else:
            self.solve(self.vv, dat2=None, verbose=verbose)

    def write_csv_driver(self, df, fname, ext):
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
        with open(fname + ext, "w") as ofile:
            ofile.write("#\n")
            df.to_csv(ofile, index=False, header=True, mode="a")

    def write_solv(self, name, atoms1, atoms2, solution, r):
        """Write solvent-solvent data to .csv file

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing correlation functions to output
        """
        all_sites = []
        gr = pd.DataFrame(r, columns=["r"])
        cr = pd.DataFrame(r, columns=["r"])
        tr = pd.DataFrame(r, columns=["r"])
        for i, j in np.ndindex(len(atoms1), len(atoms2)):
            lbl1 = atoms1[i].atom_type
            lbl2 = atoms2[j].atom_type
            gr[lbl1 + "-" + lbl2] = solution.correlations.gr[:, i, j]
            cr[lbl1 + "-" + lbl2] = solution.correlations.cr[:, i, j]
            tr[lbl1 + "-" + lbl2] = solution.correlations.tr[:, i, j]
        self.write_csv_driver(gr, name, "_driver.gvv")
        self.write_csv_driver(cr, name, "_driver.cvv")
        self.write_csv_driver(tr, name, "_driver.tvv")

    def write_solu(self, name, atoms1, atoms2, solution, r):
        """Write solvent-solvent data to .csv file

        Parameters
        ----------
        dat: Core.RISM_Obj
            Dataclass containing correlation functions to output
        """
        all_sites = []
        gr = pd.DataFrame(r, columns=["r"])
        cr = pd.DataFrame(r, columns=["r"])
        tr = pd.DataFrame(r, columns=["r"])
        for i, j in np.ndindex(len(atoms1), len(atoms2)):
            lbl1 = atoms1[i].atom_type
            lbl2 = atoms2[j].atom_type
            gr[lbl1 + "-" + lbl2] = solution.correlations.gr[:, i, j]
            cr[lbl1 + "-" + lbl2] = solution.correlations.cr[:, i, j]
            tr[lbl1 + "-" + lbl2] = solution.correlations.tr[:, i, j]
        self.write_csv_driver(gr, name, "_driver.guv")
        self.write_csv_driver(cr, name, "_driver.cuv")
        self.write_csv_driver(tr, name, "_driver.tuv")

    def read_input(self):
        """Reads .toml input file, populates vv and uv dataclasses
        and properly initialises the appropriate potentials, solvers,
        closures, and integral equations"""
        inp = toml.load(self.fname)
        name = os.path.basename(self.fname).split(sep=".")[0]
        if "solvent" not in inp:
            raise ("no solvent data in input .toml file")

        temp = inp["system"]["temp"]
        kt = inp["system"]["kT"]
        ku = inp["system"]["kU"]
        amph = inp["system"]["charge_coeff"]
        drism_damping = None
        dielec = None
        if inp["params"]["IE"] == "DRISM":
            drism_damping = inp["params"]["adbcor"]
            dielec = inp["params"]["diel"]
        nsv = inp["solvent"]["nsv"]
        nsu = None
        nspv = inp["solvent"]["nspv"]
        nspu = None
        npts = inp["system"]["npts"]
        radius = inp["system"]["radius"]
        lam = inp["system"]["lam"]
        preconv = None
        if "preconverged" in inp["solvent"]:
            preconv = inp["solvent"]["preconverged"]
        solv_atoms, solv_species = self.add_species_to_list(
            list(
                {
                    k: inp["solvent"][k]
                    for k in inp["solvent"].keys() - {"nsv", "nspv"}
                }.items()
            )
        )
        solu_atoms = None
        solu_species = None
        if "solute" in inp:
            nsu = inp["solute"]["nsu"]
            nspu = inp["solute"]["nspu"]
            solu_atoms, solu_species = self.add_species_to_list(
                list(
                    {
                        k: inp["solute"][k]
                        for k in inp["solute"].keys() - {"nsu", "nspu"}
                    }.items()
                )
            )
        data_config = DataConfig(
            temp,
            kt,
            ku,
            amph,
            drism_damping,
            dielec,
            nsv,
            nsu,
            nspv,
            nspu,
            npts,
            radius,
            lam,
            preconv,
            solv_atoms,
            solu_atoms,
            solv_species,
            solu_species,
        )
        operator_config = OperatorConfig(inp["params"]["IE"], inp["params"]["closure"])

        potential_config = PotentialConfig(
            inp["params"]["potential"], "COU", "NGR", "NGK"
        )

        solver = inp["params"]["solver"]
        picard_damping = inp["params"]["picard_damping"]
        max_iter = inp["params"]["itermax"]
        tolerance = inp["params"]["tol"]
        mdiis_settings = gillan_settings = None
        if solver == "MDIIS":
            if "mdiis_settings" not in inp["params"]:
                mdiis_settings = MDIISSettings(
                    inp["params"]["depth"], inp["params"]["picard_damping"]
                )
            else:
                mdiis_settings = MDIISSettings(
                    inp["params"]["depth"], inp["params"]["mdiis_damping"]
                )
        elif solver == "Gillan":
            gillan_settings = GillanSettings(inp["params"]["nbasis"])

        settings = SolverSettings(
            picard_damping,
            max_iter,
            tolerance,
            gillan_settings=gillan_settings,
            mdiis_settings=mdiis_settings,
        )

        solver_config = SolverConfig(solver, settings)

        # rism_job = RISMDriver(
        #     name, data_config, operator_config, potential_config, solver_config
        # )
        # solutions = rism_job.do_rism("quiet", False)
        #
        # grid = Core.Grid(data_config.npts, data_config.radius)
        #
        # self.write_solv(
        #     name,
        #     data_config.solvent_atoms,
        #     data_config.solvent_atoms,
        #     solutions.vv,
        #     grid.ri,
        # )        # literature route
        #
        # if solutions.uv:
        #     self.write_solu(
        #         name,
        #         data_config.solute_atoms,
        #         data_config.solvent_atoms,
        #         solutions.uv,
        #         grid.ri,
        #     )
        #
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
            solv_species = list(inp["solvent"].items())[2 : self.vv.nsp1 + 2]
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
                self.IE = IE(
                    self.vv, inp["params"]["diel"], inp["params"]["adbcor"], self.uv
                )
            else:
                self.IE = IE(self.vv, inp["params"]["diel"], inp["params"]["adbcor"])
        elif inp["params"]["IE"] == "XRISM-DB":
            IE = IntegralEquations.IntegralEquation(inp["params"]["IE"]).get_IE()
            if self.uv_check:
                self.IE = IE(inp["params"]["B"], self.vv, self.uv)
            else:
                self.IE = IE(inp["params"]["B"], self.vv)
            self.closure = Closures.Closure("r" + inp["params"]["closure"])
        else:
            IE = IntegralEquations.IntegralEquation(inp["params"]["IE"]).get_IE()
            if self.uv_check:
                self.IE = IE(self.vv, self.uv)
            else:
                self.IE = IE(self.vv)

        slv = Solvers.Solver(inp["params"]["solver"]).get_solver()
        slv = Solvers.Solver(inp["params"]["solver"]).get_solver()
        if inp["params"]["solver"] == "MDIIS":
            if "mdiis_damping" in inp["params"]:
                self.solver = slv(
                    self.vv,
                    inp["params"]["tol"],
                    inp["params"]["itermax"],
                    inp["params"]["picard_damping"],
                    m=inp["params"]["depth"],
                    mdiis_damping=inp["params"]["mdiis_damping"],
                )
                if self.uv_check:
                    slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                    self.solver_UV = slv_uv(
                        self.vv,
                        inp["params"]["tol"],
                        inp["params"]["itermax"],
                        inp["params"]["picard_damping"],
                        data_uv=self.uv,
                        m=inp["params"]["depth"],
                        mdiis_damping=inp["params"]["mdiis_damping"],
                    )

            else:
                self.solver = slv(
                    self.vv,
                    inp["params"]["tol"],
                    inp["params"]["itermax"],
                    inp["params"]["picard_damping"],
                    m=inp["params"]["depth"],
                    mdiis_damping=inp["params"]["picard_damping"],
                )
                if self.uv_check:
                    slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                    self.solver_UV = slv_uv(
                        self.vv,
                        inp["params"]["tol"],
                        inp["params"]["itermax"],
                        inp["params"]["picard_damping"],
                        data_uv=self.uv,
                        m=inp["params"]["depth"],
                    )
        elif inp["params"]["solver"] == "Gillan":
            self.solver = slv(
                self.vv,
                inp["params"]["tol"],
                inp["params"]["itermax"],
                inp["params"]["picard_damping"],
                nbasis=inp["params"]["nbasis"],
            )
            if self.uv_check:
                slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                self.solver_UV = slv_uv(
                    self.vv,
                    inp["params"]["tol"],
                    inp["params"]["itermax"],
                    inp["params"]["picard_damping"],
                    data_uv=self.uv,
                    nbasis=inp["params"]["nbasis"],
                )
        else:
            self.solver = slv(
                self.vv,
                inp["params"]["tol"],
                inp["params"]["itermax"],
                inp["params"]["picard_damping"],
            )
            if self.uv_check:
                slv_uv = Solvers.Solver(inp["params"]["solver"]).get_solver()
                self.solver_UV = slv_uv(
                    self.vv,
                    inp["params"]["tol"],
                    inp["params"]["itermax"],
                    inp["params"]["picard_damping"],
                    data_uv=self.uv,
                )

    def add_species_to_list(self, solv_species):
        """Parses interaction sites and assigns them to relevant species

        Parameters
        ----------
        spec_dat: List
            Contains information on the current species pulled from the .toml file
        data_object: Core.RISM_Obj
            The dataclass to which species are being assigned
        """
        atom_list = []
        species_list = []
        for spec_dat in solv_species:
            new_spec = Core.Species(spec_dat[0])
            spdict = spec_dat[1]
            new_spec.set_density(spdict["dens"])
            new_spec.set_numsites(spdict["ns"])
            site_info = list(spdict.items())[2 : new_spec.ns + 2]
            for i in site_info:
                atom = Core.Site(i[0], i[1][0], np.asarray(i[1][1]))
                new_spec.add_site(atom)
                atom_list.append(atom)
            species_list.append(new_spec)

        return (atom_list, species_list)

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
                    dat1.u[:, i, j] = sr_pot(dat2.grid.ri, i_sr_params, lam) + cou(
                        dat2.grid.ri, qi, qj, lam, dat2.amph
                    )
                else:
                    mixed = mix(i_sr_params, j_sr_params)
                    dat1.u[:, i, j] = sr_pot(dat2.grid.ri, mixed, lam) + cou(
                        dat2.grid.ri, qi, qj, lam, dat2.amph
                    )

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
                dat1.ur_lr[:, i, j] = erfr(
                    dat2.grid.ri, qi, qj, damping, 1.0, lam, dat2.amph
                )
                dat1.uk_lr[:, i, j] = erfk(
                    dat2.grid.ki, qi, qj, damping, lam, dat2.amph
                )

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
        with open(fname + ext, "w") as ofile:
            if SFE is not None:
                ofile.write(
                    "# density: {p}, temp: {T}, HNC: {HNC}, GF: {GF}, KH: {KH}, PW: {PW}, PC+: {PC_PLUS}\n".format(
                        p=p[0][0],
                        T=T,
                        HNC=SFE["HNC"],
                        GF=SFE["GF"],
                        KH=SFE["KH"],
                        PW=SFE["PW"],
                        PC_PLUS=SFE["PC+"],
                    )
                )
            else:
                ofile.write("# density: {p}, temp: {T}\n".format(p=p[0][0], T=T))
            df.to_csv(ofile, index=False, header=True, mode="a")

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
            gr[lbl1 + "-" + lbl2] = dat.g[:, i, j]
            cr[lbl1 + "-" + lbl2] = dat.c[:, i, j]
            tr[lbl1 + "-" + lbl2] = dat.t[:, i, j]
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
                gr[lbl1 + "-" + lbl2] = uv.g[:, i, j]
                cr[lbl1 + "-" + lbl2] = uv.c[:, i, j]
                tr[lbl1 + "-" + lbl2] = uv.t[:, i, j]
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
            output_str = """\n-- pyRISM --\nRunning: {name},\nTemperature: {T},\nSolvent Density: {p}\nMethod: {IE}\nClosure: {clos}\nPotential: {pot},
            """.format(
                name=self.name,
                T=str(dat1.T),
                p=str(dat1.p[0][0]),
                IE=self.IE.__class__.__name__,
                clos=self.closure.get_closure().__name__,
                pot=self.pot.get_potential()[0].__name__,
            )
            print(output_str)
        if self.uv_check:
            fuv = np.exp(-dat2.B * dat2.u_sr) - 1.0
        for j in range(1, dat1.nlam + 1):
            lam = 1.0 * j / dat1.nlam
            if j == 1:
                dat1.c = np.zeros_like(dat1.ur_lr)
                if self.uv_check:
                    dat2.c = np.zeros_like(dat2.ur_lr)
                else:
                    pass

            self.build_Ur(dat1, dat1, lam)
            self.build_renorm(dat1, dat1, 1.0, lam)
            dat1.u_sr = dat1.u - dat1.ur_lr
            # rustrism = RISMDriver(
            #     dat1.T,
            #     dat1.kT,
            #     dat1.amph,
            #     dat1.ns1,
            #     dat1.ns2,
            #     dat1.npts,
            #     dat1.radius,
            #     dat1.nlam,
            #     dat1.u,
            #     dat1.u_sr,
            #     dat1.ur_lr,
            #     dat1.uk_lr,
            #     dat1.w,
            #     dat1.p,
            #     self.solver.m,
            #     self.solver.mdiis_damping,
            #     self.solver.damp_picard,
            #     self.solver.max_iter,
            #     self.solver.tol,
            # )
            # rustrism.do_rism()
            self.solve_vv(lam, verbose)
            # dat1.c, dat1.t, dat1.h, dat1.h_k = rustrism.extract()
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
        # SFED_RBC = Functionals.Functional("RBC").get_functional()(dat2, vv)

        SFE_HNC = self.integrate(SFED_HNC, dat2.grid.d_r)
        SFE_KH = self.integrate(SFED_KH, dat2.grid.d_r)
        SFE_GF = self.integrate(SFED_GF, dat2.grid.d_r)
        SFE_SC = self.integrate(SFED_SC, dat2.grid.d_r)
        SFE_PW = self.integrate(SFED_PW, dat2.grid.d_r)
        # SFE_RBC = self.integrate(SFED_RBC, dat2.grid.d_r)

        # SFE_text = "\n{clos_name}: {SFE_val} kcal/mol"

        # print(SFE_text.format(clos_name="KH", SFE_val=SFE_KH))
        # print(SFE_text.format(clos_name="HNC", SFE_val=SFE_HNC))
        # print(SFE_text.format(clos_name="GF", SFE_val=SFE_GF))

        self.SFED = {
            "HNC": SFED_HNC,
            "KH": SFED_KH,
            "GF": SFED_GF,
            "SC": SFED_SC,
            "PW": SFED_PW,
        }
        self.SFE = {
            "HNC": SFE_HNC,
            "KH": SFE_KH,
            "GF": SFE_GF,
            "SC": SFE_SC,
            "PW": SFE_PW,
        }

        SFE_PC_PLUS = self.pc_plus()
        self.SFE["PC+"] = SFE_PC_PLUS

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

    def isothermal_compressibility(self, dat):
        # WIP
        ck = np.zeros((dat.npts, dat.ns1, dat.ns2), dtype=np.float64)

        ck = dat.grid.dht(dat.c)

        # Amber route to isothermal compressibility
        ck0 = 0.0
        for i in range(dat.grid.npts):
            ck0r = 0.0
            for j in range(0, dat.ns1):
                for k in range(j, dat.ns2):
                    if j == k:
                        msym = 1.0
                    else:
                        msym = 2.0
                    ck0r += (
                        msym * np.diag(dat.p)[j] * np.diag(dat.p)[k] * dat.c[i, j, k]
                    )
            ck0 += ck0r * dat.grid.ri[i] ** 2.0
        ck0 *= 4.0 * np.pi * dat.grid.d_r

        r = self.vv.grid.ri[:, np.newaxis, np.newaxis]
        ck0 = self.integrate(self.vv.c * r * r, 4.0 * np.pi * self.vv.grid.d_r)

        # literature route
        pck = np.sum(ck[0, ...]) * dat.p[0][0]
        p = dat.p[0][0]

        B = 1.0 / dat.T / 1.380658e-23

        # return ( 1.0 / ((dat.p[0][0] - ck0)) , 1.0 / (p * (1.0 - pck)))
        return 1.0 / (p * (1.0 - pck))
        # return 1.0 / ((dat.p[0][0] - ck0))

    def kb_partial_molar_volume(self):
        uv = self.uv
        vv = self.vv

        ck = np.zeros((uv.npts, uv.ns1, uv.ns2), dtype=np.float64)
        hk_vv = np.zeros((vv.npts, vv.ns1, vv.ns2), dtype=np.float64)
        hk_uv = np.zeros((uv.npts, uv.ns1, uv.ns2), dtype=np.float64)

        hk_vv = vv.grid.dht((vv.t + vv.c))

        ck = uv.grid.dht(uv.c)
        hk_uv = uv.grid.dht((uv.t + uv.c))
        # hk_vv = self.vv.h_k
        # hk_uv = self.uv.h_k

        compres = self.isothermal_compressibility(self.vv)

        r = self.uv.grid.ri[:, np.newaxis, np.newaxis]
        ck0 = self.integrate(self.uv.c * r * r, 4.0 * np.pi * self.uv.grid.d_r)
        rhvv = self.integrate(self.vv.h * r * r, 4.0 * np.pi * self.uv.grid.d_r)
        rhuv = self.integrate(self.uv.h * r * r, 4.0 * np.pi * self.uv.grid.d_r)
        khvv = np.sum(hk_vv[0, ...])
        khuv = np.sum(hk_uv[0, ...])
        pv = self.vv.p[0][0]
        pvec = np.diag(self.vv.p)

        inv_B = self.uv.kT * self.uv.T
        ck0_direct = np.sum(ck[0, ...] @ self.vv.p)

        return (1.0 / pv) + (khvv - khuv) / self.uv.ns1

    def rism_kb_partial_molar_volume(self):
        uv = self.uv

        ck = np.zeros((uv.npts, uv.ns1, uv.ns2), dtype=np.float64)
        hk = np.zeros((uv.npts, uv.ns1, uv.ns2), dtype=np.float64)

        ck = uv.grid.dht(uv.c)
        hk = self.uv.h_k

        compres = self.isothermal_compressibility(self.vv)

        r = self.uv.grid.ri[:, np.newaxis, np.newaxis]
        ck0 = self.integrate(self.uv.c * r * r, 4.0 * np.pi * self.uv.grid.d_r)
        rhvv = self.integrate(self.vv.h * r * r, 4.0 * np.pi * self.uv.grid.d_r)
        rhuv = self.integrate(self.uv.h * r * r, 4.0 * np.pi * self.uv.grid.d_r)
        khvv = np.sum(hk[0, 0, 0])
        khuv = np.sum(hk[0, :, 0])
        pv = self.vv.p[0][0]
        pvec = np.diag(self.vv.p)

        inv_B = self.uv.kT * self.uv.T
        ck0_direct = np.sum(ck[0, ...])

        return compres * (1.0 - pv * ck0_direct)

        # return (1.0 / pv) + khvv - khuv / self.uv.ns1

    def dimensionless_pmv(self):
        pmv = self.kb_partial_molar_volume()

        return self.uv.p[0][0] * pmv

    def pc_plus(self):
        pc, pcplus = self.pressure()
        pmv = self.kb_partial_molar_volume()

        if self.closure.get_closure().__name__ == "HyperNetted_Chain":
            keystr = "HNC"
        elif self.closure.get_closure().__name__ == "KovalenkoHirata":
            keystr = "KH"
        else:
            keystr = "GF"

        return self.SFE[keystr] - (pcplus * pmv)

    def __virial_pressure(self):
        # WIP
        duvv = np.zeros_like(self.vv.u)

        for i, j in np.ndindex(self.vv.ns1, self.vv.ns2):
            grad = np.gradient(self.vv.u[:, i, j], self.vv.grid.d_r)
            duvv[:, i, j] = grad

        pressure = (self.vv.g @ duvv) * np.power(self.uv.grid.ri, 3)[
            :, np.newaxis, np.newaxis
        ]

        return (
            self.vv.kT * self.vv.T * np.sum(self.vv.p)
        ) - self.vv.grid.d_r * 2.0 / 3.0 * np.pi * np.power(
            np.sum(self.vv.p), 2
        ) * np.sum(
            pressure
        )

    def pressure(self):
        uv = self.vv
        nu = uv.ns1
        nv = uv.ns2

        p0 = uv.p[0][0]

        B = 1.0 / uv.T / 1.380658e-23

        inv_B = uv.T * 1.380658e-23

        inv_py_B = 1.0 / uv.B

        ck = np.zeros((uv.npts, uv.ns1, uv.ns2), dtype=np.float64)

        ck = uv.grid.dht(uv.c)

        compres = self.isothermal_compressibility(uv)

        initial_term = ((nu + 1.0) / 2.0) * (uv.kU * uv.T) * p0

        ck_direct = np.sum(uv.p @ uv.p @ ck[0, ...])

        ck_compres = np.power(p0, 2) * (1.0 - (uv.B / compres))

        pressure = np.sum(initial_term) - ((uv.kU * uv.T) / 2.0) * ck_direct

        ideal_pressure = p0 * uv.kU * uv.T

        return pressure, pressure - ideal_pressure


@jit
def build_Ur_impl(
    npts, ns1, ns2, sr_pot, mix, cou, atoms1, atoms2, r, charge_coeff, lam=1
):
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
                u[:, i, j] = sr_pot(r, i_sr_params, lam) + cou(
                    r, qi, qj, lam, charge_coeff
                )
            else:
                mixed = mix(i_sr_params, j_sr_params)
                u[:, i, j] = sr_pot(r, mixed, lam) + cou(r, qi, qj, lam, charge_coeff)
        return u