#!/usr/bin/env python

import sys
sys.path.append('..')

from pathlib import Path
from rism_ctrl import *

if __name__ == "__main__":
    pathlist = list(Path('../data').resolve().rglob('*.toml'))
    output_path = Path('./outputs').resolve()
    name = [path.stem for path in list(pathlist)]
    f = open("pyrism_outputs.log", 'w')
    for i, path in enumerate(pathlist):
        toml_file = str(path)
        print(toml_file)
        (Path(output_path) / Path(name[i])).mkdir(parents=True, exist_ok=True)
        os.chdir((Path(output_path) / Path(name[i])).resolve())
        print(os.getcwd())
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
        os.chdir(output_path)
    f.close()

