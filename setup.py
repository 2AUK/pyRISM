from setuptools import setup, find_packages

setup(
        name='pyRISM',
        version='0.1.0',
        packages=find_packages(where='pyrism/', include=['pyrism', 'pyrism/Closures', 'pyrism/Solvers', 'pyrism/IntegralEquations', 'pyrism/Core', 'pyrism/Functionals']),
        package_dir = {
            'pyrism': 'pyrism',
            'pyrism.Closures': 'pyrism/Closures',}
            )
