#!/usr/bin/env python

import sys
sys.path.append('..')

from pathlib import Path
from rism_ctrl import *


pathlist = Path('../data').resolve().rglob('*.toml')
f = open("pyrism_outputs.log", 'a')
print(list(pathlist))
for path in pathlist:
    toml_file = str(path)
    print(toml_file)
    try:
        mol = RismController(toml_file)
        mol.initialise_controller()
    except Exception as e:
        f.write("{file} not successful due to error: {e}\n".format(file=toml_file, e=e))
        f.flush()
        continue
    mol.write_check = True
    try:
        mol.do_rism()
    except:
        f.write("{file} did not converge\n".format(file=toml_file))
        f.flush()
        continue
f.close()
