from pyrism.rism_ctrl import *
import unittest
import numpy as np

class TestRISMJobs(unittest.TestCase):

    def test_2_propanol(self):
        mol = RismController("inputs/2_propanol.toml")
        mol.initialise_controller()
        #mol.do_rism()

        test_case = np.genfromtxt('outputs/2_propanol/2_propanol.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.data_uv.grid.npts, mol.data_uv.ns1, mol.data_uv.ns2))

        print(test_case)
        print(test_case.shape)

unittest.main()        


