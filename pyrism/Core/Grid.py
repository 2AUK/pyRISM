#!/usr/bin/env python3
"""
grid.py
Defines Grid object to handle generation and transforms

"""
from dataclasses import dataclass, field
import numpy as np
from scipy.fftpack import dst, idst
from .Transforms import discrete_hankel_transform, inverse_discrete_hankel_transform


@dataclass(init=True)
class Grid:
    npts: int
    radius: float
    ri: np.ndarray = field(init=False)
    ki: np.ndarray = field(init=False)
    d_r: float = field(init=False)
    d_k: float = field(init=False)

    def __post_init__(self):
        self.ri = np.zeros(self.npts, dtype=float)
        self.ki = np.zeros(self.npts, dtype=float)
        self.d_r = self.radius / float(self.npts)
        self.d_k = 2 * np.pi / (2 * float(self.npts) * self.d_r)
        self.generate_grid()

    def generate_grid(self):
        """
        Generates r-space and k-space grids to compute functions over

        Parameters
        ----------

        Returns
        -------

        ri: ndarray
           r-space grid
        ki: ndarry
           k-space grid
        """

        for i in np.arange(0, int(self.npts)):
            self.ri[i] = (i + 0.5) * self.d_r
            self.ki[i] = (i + 0.5) * self.d_k

    def dht(self, fr: np.ndarray) -> np.ndarray:
        """
        Discrete Hankel Transform

        Parameters
        ----------

        fr: ndarray
           Function to be transformed from r-space to k-space

        Returns
        -------

        fk: ndarray
           Transformed function from r-space to k-space
        """
        return discrete_hankel_transform(self.ri, self.ki, fr, self.d_r)

    def idht(self, fk: np.ndarray) -> np.ndarray:
        """
        Inverse Discrete Hankel Transform

        Parameters
        ----------

        fk: ndarray
           Function to be transformed from k-space to r-space

        Returns
        -------

        fr: ndarray
           Transformed function from k-space to r-space
        """
        return inverse_discrete_hankel_transform(self.ri, self.ki, fk, self.d_k)
