# Sample Data

This directory contains various inputs for the the pyrism code

## Format
```
[system]
temp = 300 #System temperature
kT = 1.0 #Value for Boltzmann constant 
charge_coeff = 167101.0 #Constant to get correct units for coulomb potential
npts = 2048 #Number of points
radius = 20.48 #Radius
solver = "Ng" #Solver (non-functional right now)
picard_damping = 0.001 #Damping parameter for the mixing procedure in picard iteration
itermax = 10000 #Max number of iterations
lam = 10 #Max number of charging cycles 
tol = 1E-7 #Tolerance for normed RMSE 
closure = "KH" #Current closure available: HNC, PY, KH, PSE-n, KGK 
```

The system section sets up the thermodynamic state as well as defining the solver parameters.

```
[solvent]
nsv = 3 #Number of solvent sites

"O" = [
    [78.15, 3.16572, -0.8476, 0.033314], #Forcefield parameters in order: epsilon, sigma, q, density
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00] #Coordinates: x, y, z
]

"H1" = [
    [7.815, 1.16572, 0.4238, 0.033314], #Forcefield parameters in order: epsilon, sigma, q, density
    [1.00000000e+00, 0.00000000e+00, 0.00000000e+00] #Coordinates: x, y, z
] 

"H2" = [
    [7.815, 1.16572, 0.4238, 0.033314], #Forcefield parameters in order: epsilon, sigma, q, density
    [-3.33314000e-01, 9.42816000e-01, 0.00000000e+00] #Coordinates: x, y, z
]
```

The solvent section defines the forcefield parameters and coordinates of the solvent. Only works for single solvent systems right now, not mixtures.

```
[solute]
nsu = 0
```

The solute section defines the forcefield parameters and coordinates of the solute. Currently not implemented

<!--- 
## Including package data

Modify your package's `setup.py` file and the `setup()` command. Include the 
[`package_data`](http://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use) keyword and point it at the 
correct files.
--->
## Manifest

* `cSPCE.toml`: cSPCE water model taken from the [AMBER software package](https://ambermd.org/index.php)

* `HR1982.toml`: Charged liquid nitrogen [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)

* `HR1982N.toml`: Neutral liquid nitrogen [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)

* `HR1982_HCl_II.toml`: Hydrochloric acid model II [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)

* `HR1982_HCl_III.toml`: Hydrochloric acid model III [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)

* `HR1982_Br2_I.toml`: Bromine model I [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)

* `HR1982_Br2_III.toml`: Bromine model III [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)

* `HR1982_Br2_IV.toml`: Bromine model IV [Application of an extended RISM equation to dipolar and quadrupolar fluids](https://aip.scitation.org/doi/abs/10.1063/1.443606)