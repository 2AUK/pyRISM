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
- Coulomb Potential
- Long-Range Coulomb Potential

#### Closures
- Hyper-Netted Chain (HNC)
- Percus-Yevick (PY)
- Kovalenko-Hirata (KH)
- Partial Series Expansion (PSE-n)

#### Solvers
- Picard Iteration
- Ng-accelerated convergence
- Newton-Krylov (from Scipy)
- Anderson (from Scipy)

### Usage
Run
`$ python rism_ctrl.py path/to/file`

Various exemplar inputs are given in `data/`

### Future Work
- DRISM
- Solute-solvent interactions
- More potentials (Hard-sphere, AB-form LJ)
- MDIIS solver
- Chemical potentials

### Copyright

Copyright (c) 2021, Abdullah Ahmad


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
