#!/usr/bin/env python
import subprocess
import tempfile
import sys
from pathlib import Path
import os
import toml

inp_dir = Path(sys.argv[1])

for toml_file in inp_dir.rglob("*.toml"):
    print(toml_file)
    inp_toml = toml.load(toml_file)
    inp_toml["system"]["temp"] = 300
    inp_toml["system"]["npts"] = 8192
    inp_toml["system"]["radius"] = 40.96
    inp_toml["params"]["solver"] = "Ng"
    inp_toml["params"]["closure"] = "KH"
    #inp_toml["params"]["IE"] = "XRISM"
    inp_toml["params"]["picard_damping"] = 0.5
    inp_toml["params"]["mdiis_damping"] = 0.1
    inp_toml["params"]["depth"] = 20
    inp_toml["params"]["tol"] = 1e-12
    # inp_toml["solvent"][
    #         "preconverged"
    #     ] = "/users/tjb20156/pyRISM/data/benchmark/solvent_data/solvent_bins/chloroform_Extended_RISM_Kovalenko-Hirata_298.15K.bin"
    with open(toml_file, "w") as tfile:
        toml.dump(inp_toml, tfile, encoder=toml.TomlNumpyEncoder())
