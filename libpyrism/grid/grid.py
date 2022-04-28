#!/usr/bin/env python3

from dataclasses import dataclass, field
import numpy as np

@dataclass(init=True)
class Grid(object):
    """
    Grid object - all functions are tabulated with Grid
    A midpoint grid is generated using the number of points `npts` and
    interval [0, `radius`].
    DST Type IV is use for transforms on the grid.

    Attributes
    ----------
    npts: int
    Number of points in grid
    radius: float
    Radius of grid (max distance in Angstroms)
    r_grid: np.ndarray
    Tabulated grid in r-space
    k_grid: np.darray
    Tabulated grid in k-space
    dr: float
    Grid spacing in r-space
    dk: float
    Grid spacing in k-space
    """

    npts: int
    radius: float
    r_grid: np.ndarray = field(init=False)
    k_grid: np.ndarray = field(init=False)
    dr: float = field(init=False)
    dk: float = field(init=False)

    def __post_init__(self):
        """Calculate grid parameters and generate arrays post class initalization"""
        self.dr = self.radius / float(self.npts)
        self.dk = 2 * np.pi / (2 * float(self.npts) * self.dr)
        self.r_grid, self.k_grid = self.tabulate_grid()


    def tabulate_grid(self):
        """Tabulates r-space and k-space grids

        Returns
        -------
        r_grid: np.ndarray
        Tabulated grid in r-space
        k_grid: np.darray
        Tabulated grid in k-space
        """
        r_grid = np.arange(0.5, self.npts) * self.dr
        k_grid = np.arange(0.5, self.npts) * self.dk

        return r_grid, k_grid

if __name__ == "__main__":
    new_grid = Grid(16384, 20.48)
    print(np.arange(0.5, 16384))
    print(new_grid.r_grid)
