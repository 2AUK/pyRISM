import attr
import numpy as np


@attr.s
class CalculationData(object):

    # Initial parameters required to instantiate the other attributes
    T: float = attr.ib()
    kT: float = attr.ib()
    e: float = attr.ib()
    lam: int = attr.ib()
    ns1: int = attr.ib()
    ns2: int = attr.ib()
    npts: int = attr.ib()

    # Set of attributes that are iterated during the RISM calculation
    c: np.ndarray = attr.ib(init=False)
    t: np.ndarray = attr.ib(init=False)
    h: np.ndarray = attr.ib(init=False)

    # Set of attributes that remain constant during the RISM calculation
    B: float = attr.ib(init=False)
    u: np.ndarray = attr.ib(init=False)
    u_sr: np.ndarray = attr.ib(init=False)
    u_lr: np.ndarray = attr.ib(init=False)
    w: np.ndarray = attr.ib(init=False)
    p: np.ndarray = attr.ib(init=False)

    def __attrs_post_init__(self):

        self.B = 1 / self.T / self.kT
        self.c = self.t = self.h = self.u = self.u_sr = self.u_lr = self.w = np.zeros(
            (self.npts, self.ns1, self.ns2), dtype=np.float64
        )
        self.p = np.zeros((self.ns1, self.ns2), dtype=np.float64)