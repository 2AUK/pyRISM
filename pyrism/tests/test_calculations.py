from pyrism.rism_ctrl import *
import unittest
from numpy.testing import assert_array_almost_equal
import warnings
from pathlib import Path

class TestRISMJobs(unittest.TestCase):

    def setUp(self):
        self.verbosity = False 
        warnings.simplefilter('ignore')

    def test_2_propanol(self):
        mol = RismController(Path("./inputs/2_propanol.toml").resolve())
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('./outputs/2_propanol/2_propanol.guv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        assert_array_almost_equal(test_case, mol.uv.g)

    def test_argon_uv(self):
        mol = RismController("inputs/argon_uv.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/argon_uv/argon_uv.tuv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        assert_array_almost_equal(test_case, mol.uv.t)

    def test_chg_org_solu(self):
        mol = RismController("inputs/chg_org_solu.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/chg_org_solu/chg_org_solu.cuv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.uv.grid.npts, mol.uv.ns1, mol.uv.ns2))

        assert_array_almost_equal(test_case, mol.uv.c)

    def test_chloro(self):
        mol = RismController("inputs/chloro.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/chloro/chloro_SFED.duv', delimiter=',', skip_header=2, usecols=1)

        assert_array_almost_equal(test_case, mol.SFED['HNC'])

    def test_chloro_cg(self):
        mol = RismController("inputs/chloro_cg.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/chloro_cg/chloro_cg_SFED.duv', delimiter=',', skip_header=2, usecols=2)

        assert_array_almost_equal(test_case, mol.SFED['KH'])

    def test_nhexylbenzene(self):
        mol = RismController("inputs/nhexylbenzene.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/nhexylbenzene/nhexylbenzene_SFED.duv', delimiter=',', skip_header=2, usecols=3)

        assert_array_almost_equal(test_case, mol.SFED['GF'])
    
    def test_cSPCE(self):
        mol = RismController("inputs/cSPCE.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/cSPCE/cSPCE.gvv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.vv.grid.npts, mol.vv.ns1, mol.vv.ns2))

        assert_array_almost_equal(test_case, mol.vv.g)

    def test_cSPCE_NaCl(self):
        mol = RismController("inputs/cSPCE_NaCl.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/cSPCE_NaCl/cSPCE_NaCl.tvv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.vv.grid.npts, mol.vv.ns1, mol.vv.ns2))

        assert_array_almost_equal(test_case, mol.vv.t)

    def test_HR1982_Br2_IV(self):
        mol = RismController("inputs/HR1982_Br2_IV.toml")
        mol.initialise_controller()
        mol.do_rism(verbose=self.verbosity)

        test_case = np.genfromtxt('outputs/HR1982_Br2_IV/HR1982_Br2_IV.cvv', delimiter=',', skip_header=2)[:, 1:].reshape((mol.vv.grid.npts, mol.vv.ns1, mol.vv.ns2))

        assert_array_almost_equal(test_case, mol.vv.c)

if __name__ == "__main__":
    unittest.main()        


