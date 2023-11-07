pyRISM
==============================
[//]: # (Badges)
[![DOI](https://zenodo.org/badge/267991398.svg)](https://zenodo.org/badge/latestdoi/267991398)

A Rust implementation of the RISM equations

### Currently Implemented Features
- XRISM and DRISM for neat liquids, mixtures, and solute-solvent systems
- Picard, Ng, and MDIIS Solvers (more to come!)
- Compressed solvent-solvent solutions (for solving subsequent solute-solvent problem)
- Solvation free energy densities
- Thermodynamical data (Partial Molar Volume, Isothermal Compressibility, etc.)
- Arbitrary inputs can be defined (though no guarantee everything will converge...)

#### Potentials
- Lennard-Jones Potential
- Coulomb Potential
- Long-Range Coulomb Potential

#### Closures
- Hyper-Netted Chain (HNC)
- Percus-Yevick (PY)
- Kovalenko-Hirata (KH)
- Partial Series Expansion (PSE-n)

### Installation
The code requires a system installation of [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/wiki/Precompiled-installation-packages) and the [Rust](https://www.rust-lang.org/tools/install) compiler.

pyRISM can be installed with `make install`.

### Usage 
Currently, the calculator is focused for command-line usage with a very minimal library interface.
This will change soon with bindings in both Rust and Python.

The command-line tool can be called with:
`rism [OPTIONS] <input_file.toml>`

The list of options:
```
[-h|--help]      Show help message
[-c|--compress]  Compress the solvent-solvent problem for future use
[-q|--quiet]     Suppress all output from solver (DEFAULT)
[-v|--verbose]   Print basic information from solver
[-l|--loud]      Print all information from solver
```
### Future Work
- LMV Solver (in progress)
- Gillan Solver (in progress)

### Copyright

Copyright (c) 2023, Abdullah Ahmad
