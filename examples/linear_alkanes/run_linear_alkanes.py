from pyrism.librism import Calculator
from pathlib import Path
from multiprocessing import Pool

inputs = Path('.').glob('*.toml')


def run_linear_alkane_jobs(inp):
    inp = str(inp)
    s, t, g = Calculator(inp, "quiet", False).execute()

with Pool(3) as p:
    p.map(run_linear_alkane_jobs, inputs)
