from dataclasses import dataclass, field
import numpy as np
from .Grid import Grid
from .Site import Site
from .Species import Species


@dataclass
class RISM_Obj(object):

    # Initial parameters required to instantiate the other attributes
    T: float
    kT: float
    kU: float
    amph: float
    ns1: int
    ns2: int
    nsp1: int
    nsp2: int
    npts: int
    radius: float
    nlam: int

    # Set of attributes that are iterated during the RISM calculation
    c: np.ndarray = field(init=False)
    c_prev: np.ndarray = field(init=False)
    t: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)
    g: np.ndarray = field(init=False)
    h_k: np.ndarray=field(init=False)

    # Set of attributes that remain constant during the RISM calculation
    B: float = field(init=False)
    u: np.ndarray = field(init=False)
    u_sr: np.ndarray = field(init=False)
    ur_lr: np.ndarray = field(init=False)
    uk_lr: np.ndarray = field(init=False)
    w: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)
    grid: Grid = field(init=False)
    species: list = field(init=False, default_factory=list)
    atoms: list = field(init=False, default_factory=list)
    Q_r: np.ndarray = field(init=False) #XRISM-DB
    Q_k: np.ndarray = field(init=False) #XRISM-DB
    tau: np.ndarray = field(init=False) #XRISM-DB

    def calculate_beta(self):
        self.B = 1 / self.T / self.kT

    def __post_init__(self):

        self.calculate_beta()
        self.c_prev = np.zeros((self.npts, self.ns1, self.ns2), dtype=np.float64)
        self.t = np.zeros((self.npts, self.ns1, self.ns2), dtype=np.float64)
        self.h = np.zeros((self.npts, self.ns1, self.ns2), dtype=np.float64)
        self.h_k = np.zeros_like(self.h)
        self.Q_k = np.zeros_like(self.h)
        self.Q_r = np.zeros_like(self.h)
        self.tau = np.zeros_like(self.h)
        self.u = np.zeros((self.npts, self.ns1, self.ns2), dtype=np.float64)
        self.u_sr = np.zeros_like(self.u)
        self.ur_lr = np.zeros_like(self.u)
        self.uk_lr = np.zeros_like(self.u)
        self.c = np.zeros((self.npts, self.ns1, self.ns2), dtype=np.float64)
        self.w = np.zeros((self.npts, self.ns1, self.ns1), dtype=np.float64)
        self.g = np.zeros((self.npts, self.ns1, self.ns2), dtype=np.float64)
        self.p = np.zeros((self.ns1, self.ns2), dtype=np.float64)
        self.grid = Grid(self.npts, self.radius)

    def __iter__(self):
        return iter((
            self.T, 
            self.kT,
            self.kU,
            self.amph,
            self.ns1,
            self.ns2,
            self.nsp1,
            self.nsp2,
            self.npts,
            self.radius,
            self.nlam,
            self.c,
            self.t,
            self.h,
            self.h_k,
            self.g,
            self.B,
            self.u,
            self.u_sr,
            self.ur_lr,
            self.uk_lr,
            self.w,
            self.p,
            self.grid.ri,
            self.grid.ki,
        ))
