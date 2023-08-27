from pyrism.rism_ctrl import *
from pyrism import rust_helpers
import matplotlib.pyplot as plt

print(rust_helpers.sum_as_string(5, 10))

mol = RismController("../data/cSPCE_XRISM_methane.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)

print("isothermal compressibility (1/A^3):", (mol.isothermal_compressibility(mol.vv)))

pressure, pressure_plus = mol.pressure()

print("pressure (kcal/mol/A^3):", pressure)

print("pressure - ideal pressure (kcal/mol/A^3):", pressure_plus)

KB_PMV = mol.kb_partial_molar_volume()
RISM_KB_PMV = mol.rism_kb_partial_molar_volume()

print("KB PMV (A^3):", KB_PMV)

print("KB PMV (cm^3 / mol):", KB_PMV / 1e24 * 6.022E23)

print("RISM-KB PMV (A^3):", RISM_KB_PMV)

print("RISM-KB PMV (cm^3 / mol):", RISM_KB_PMV / 1e24 * 6.022E23)

print("Dimensionless PMV:", mol.dimensionless_pmv())

print("HNC (kcal/mol):", mol.SFE['HNC'])

print("PW (kcal/mol):", mol.SFE['PW'])

print("PC+ (kcal/mol):", mol.pc_plus())