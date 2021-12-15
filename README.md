pyrism
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/pyrism.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/pyrism)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/pyrism/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/pyrism/branch/master)

A pedagogical implementation of the RISM equations

### Currently Implemented Features
- XRISM and DRISM for neat liquids and solute-solvent systems

#### Potentials
- Lennard-Jones Potential
- Coulomb Potential
- Long-Range Coulomb Potential

#### Closures
- Hyper-Netted Chain (HNC)
- Percus-Yevick (PY)
- Kovalenko-Hirata (KH)
- Partial Series Expansion (PSE-n)

### Usage
Pass the input file as an argument to `rism_ctrl.py`

`python rism_ctrl.py INPUT.toml`

Example input files are provided in `pyrism\data` with references in `pyrism\data\README.md`

### Future Work
- More potentials (Hard-sphere, AB-form LJ)
- MDIIS solver

### Copyright

Copyright (c) 2021, Abdullah Ahmad


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
