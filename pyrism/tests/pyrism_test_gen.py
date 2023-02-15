#!/usr/bin/env python

import sys
sys.path.append('..')

from pathlib import Path
from rism_ctrl import *


pathlist = Path('../data').resolve().rglob('*.toml')
for path in pathlist:
    toml_file = str(path)
    print(toml_file)
    try:
        mol = RismController(toml_file)
        mol.initialise_controller()
        mol.write_check = True
        mol.do_rism()
    except:
        print("{file} did not converge".format(file=toml_file))
        continue
