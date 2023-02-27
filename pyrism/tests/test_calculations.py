from pyrism.rism_ctrl import *
import unittest
import numpy as np
import warnings

class TestRISMJobs(unittest.TestCase):

    def setUp(self):
        self.verbosity = False
        warnings.simplefilter('ignore')

    def test_2_propanol(self):
        mol = RismController("./inputs/2_propanol.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('./outputs/2_propanol/2_propanol.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        np.allclose(test_case, mol.uv.g, 1e-7, 1e-7)

    def test_argon_uv(self):
        mol = RismController("inputs/argon_uv.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/argon_uv/argon_uv.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        np.allclose(test_case, mol.uv.g, 1e-7, 1e-7)

    def test_chg_org_solu(self):
        mol = RismController("inputs/chg_org_solu.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/chg_org_solu/chg_org_solu.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        np.allclose(test_case, mol.uv.g, 1e-7, 1e-7)

    def test_chloro(self):
        mol = RismController("inputs/chloro.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/chloro/chloro.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        np.allclose(test_case, mol.uv.g, 1e-7, 1e-7)

    def test_chloro_cg(self):
        mol = RismController("inputs/chloro_cg.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/chloro_cg/chloro_cg.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        np.allclose(test_case, mol.uv.g, 1e-7, 1e-7)

    def test_cSPCE(self):
        mol = RismController("inputs/cSPCE.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/cSPCE/cSPCE.gvv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.vv.grid.npts, mol.vv.ns1, mol.vv.ns2))

        np.allclose(test_case, mol.vv.g, 1e-7, 1e-7)

    def test_cSPCE_NaCl(self):
        mol = RismController("inputs/cSPCE_NaCl.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/cSPCE_NaCl/cSPCE_NaCl.gvv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.vv.grid.npts, mol.vv.ns1, mol.vv.ns2))

        np.allclose(test_case, mol.vv.g, 1e-7, 1e-7)

    def test_HR1982N(self):
        mol = RismController("inputs/HR1982N.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/HR1982N/HR1982N.gvv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.vv.grid.npts, mol.vv.ns1, mol.vv.ns2))

        np.allclose(test_case, mol.vv.g, 1e-7, 1e-7)
if __name__ == "__main__":
    unittest.main()        


