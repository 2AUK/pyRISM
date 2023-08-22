from pyrism.rism_ctrl import *
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

mol = RismController("../data/cSPCE_XRISM_methane.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)


print(mol.SFED['KH'])