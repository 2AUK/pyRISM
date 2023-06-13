from pyrism.rism_ctrl import *
import unittest
from numpy.testing import assert_allclose, assert_almost_equal
import warnings
from pathlib import Path
 

class TestRISMBuilder(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        self.test_species = Core.Species('test')
        test_species.set_numsites(1)
        test_species.set_density(0.1)
        test_species.add_site(Core.Site('x', [120.0, 1.16, 0.0], np.asarray([0.0, 0.0, 0.0])))

    def test_Data(self):
        data = (
            DataBuilder()
            .temperature(100.0)
            .boltzmann(1.0)
            .boltzmann_energy(2.0)
            .charge_scale(167101.0)
            .num_sites1(1)
            .num_sites2(1)
            .num_species2(1)
            .num_species2(1)
            .num_points(2)
            .rad(2.0)
            .num_lam_cycles(1)
            .add_species(self.test_species)
            .build()
        )

        assert_almost_equal(0.01, data.B)