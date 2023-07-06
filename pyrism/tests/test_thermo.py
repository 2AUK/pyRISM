from pyrism.rism_ctrl import *
import matplotlib.pyplot as plt

mol = RismController("../data/cSPCE_XRISM_ethene.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)

#plt.plot(mol.vv.grid.ri, mol.uv.g[:, 0, 0])
#plt.show()

print("isothermal compressibility:", (mol.isothermal_compressibility(mol.vv)))

print("pressure:", mol.pressure())

print("PMV:", mol.partial_molar_volume())