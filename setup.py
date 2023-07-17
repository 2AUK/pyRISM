from setuptools import setup, find_packages

setup(
        name='pyRISM',
        version='0.2.0',
        packages=find_packages(),
        entry_points = {
            'console_scripts' : ['pyrism=pyrism.__main__:cli'],
            },
        package_data={"pyrism/cnn_data": ["/*"]},
            )
