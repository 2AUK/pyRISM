#!/usr/bin/env python3
from grid.grid import Grid
import pytest
import numpy as np

def test_grid():
    new_grid = Grid(10, 20.48)

    ref_r_grid = [1.024, 3.072, 5.12, 7.168, 9.216,
                  11.264, 13.312, 15.36, 17.408, 19.456]
    ref_k_grid = [0.07669904, 0.23009712, 0.3834952 , 0.53689328, 0.69029135,
                  0.84368943, 0.99708751, 1.15048559, 1.30388367, 1.45728175]
    ref_dr = 2.048
    ref_dk = 0.15339807878856412

    np.testing.assert_almost_equal(new_grid.r_grid, ref_r_grid)
    np.testing.assert_almost_equal(new_grid.k_grid, ref_k_grid)
    np.testing.assert_almost_equal(new_grid.dr, ref_dr)
    np.testing.assert_almost_equal(new_grid.dk, ref_dk)
