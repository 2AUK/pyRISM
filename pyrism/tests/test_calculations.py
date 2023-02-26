from pyrism.rism_ctrl import *
import unittest
import numpy as np

class TestRISMJobs(unittest.TestCase):

    def test_2_propanol(self):
        mol = RismController("inputs/2_propanol.toml")
        mol.initialise_controller()
        mol.do_rism()

        test_case = np.genfromtxt('outputs/2_propanol/2_propanol.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        np.allclose(test_case, mol.uv.g, 1e-7, 1e-7)


unittest.main()        


