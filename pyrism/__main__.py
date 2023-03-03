import click
from pyrism.rism_ctrl import *

@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('-T', '--temp', 'temperature', type=float, help='Change system temperature')
@click.option('-v', '--verbose', 'verbosity', is_flag=True, help='Output job and iteration information')
@click.option('-w', '--write', 'write', type=click.Choice(['duv', 'all'],  case_sensitive=False), help='Output the solvation free energy density [-w duv] or all solvent-solvent and solute-solvent files along with solvation free energy density [-w all]')
def cli(filename, temperature, verbosity, write):
    mol = RismController(filename)
    mol.initialise_controller()
    if temperature is not None:
        mol.vv.T = float(temperature)
        mol.vv.calculate_beta()
        if mol.uv_check:
            mol.uv.T = float(temperature)
            mol.uv.calculate_beta()
    try:
        mol.do_rism(verbose=verbosity)
    except RuntimeError as e:
        print(e)
    if write == 'duv':
        mol.write_output(duv_only=True)
    elif write == 'all':
        mol.write_output()
    else:
        pass
