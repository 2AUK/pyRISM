pyrism
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/pyrism.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/pyrism)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/pyrism/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/pyrism/branch/master)

A pedagogical implementation of the RISM equations

### Currently Implemented Features
- XRISM for neat liquids

#### Potentials
- Lennard-Jones Potential

### Usage
Currently the inputs are just specified in `pyrism/rism_ctrl.py`

Comment out which example you want to use and run `python rism_ctrl.py`

Example input files are provided in `pyrism\data` with references in `pyrism\data\README.md`

### Future Work
- DRISM
- Solute-solvent interactions
- More closures (PY, KH)
- More potentials (Hard-sphere, AB-form LJ)
- MDIIS solver
- Chemical potentials

### Copyright

Copyright (c) 2021, Abdullah Ahmad


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
