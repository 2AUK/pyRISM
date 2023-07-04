from pyrism.rism_ctrl import *

mol = RismController("../data/cSPCE_XRISM.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)

print(mol.isothermal_compressibility(mol.vv))
