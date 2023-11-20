import pandas as pd
import pubchempy as pcp
import sys

df = pd.read_csv(sys.argv[1])

mols = df['Name'].tolist()

for mol in mols:
    name = mol.replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')
    print(name)
    pcp.download('SDF', name + '.sdf', mol, 'name', overwrite=True)
