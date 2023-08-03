from pyrism.rism_ctrl import *
import matplotlib.pyplot as plt

mol = RismController("../data/cSPCE_XRISM_methane.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)

#plt.plot(mol.vv.grid.ri, mol.uv.c[:, 0, 0])
#plt.show()

print("isothermal compressibility (1/A^3):", (mol.isothermal_compressibility(mol.vv)))

pressure, pressure_plus = mol.pressure()

print("pressure (kcal/mol/A^3):", pressure)

print("pressure - ideal pressure (kcal/mol/A^3):", pressure_plus)

PMV = mol.partial_molar_volume()

print("PMV (A^3):", PMV)

print("PMV (cm^3 / mol):", PMV / 1e24 * 6.022E23)

print("Dimensionless PMV:", mol.dimensionless_pmv())

print("HNC (kcal/mol):", mol.SFE['HNC'])

print("PW (kcal/mol):", mol.SFE['PW'])

print("PC+ (kcal/mol):", mol.pc_plus())