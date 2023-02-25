pyrism
==============================
[//]: # (Badges)
[![DOI](https://zenodo.org/badge/267991398.svg)](https://zenodo.org/badge/latestdoi/267991398)

A Python implementation of the RISM equations

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

### Command-line Usage
Pass the input file as an argument to `rism_ctrl.py`

`python rism_ctrl.py INPUT.toml <True/False [OPTIONAL, default=False]> <Temp [OPTIONAL]>`

The second argument allows you to define whether pyRISM outputs the correlation functions to files.
The third argument allows you to override the temperature set in the `.toml` file.
Example input files are provided in `pyrism\data` with references in `pyrism\data\README.md`

### Future Work
- Gillan's method

### Copyright

Copyright (c) 2023, Abdullah Ahmad
