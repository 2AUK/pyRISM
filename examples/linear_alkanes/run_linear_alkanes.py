from pyrism.librism import Calculator
from pathlib import Path
from multiprocessing import Pool

inputs = Path('./linear_alkanes/').glob('*.toml')

for inp in inputs:
    inp = str(inp)
    s, t, g = Calculator(inp, "verbose", False).execute()
