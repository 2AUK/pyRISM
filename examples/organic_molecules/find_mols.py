import pandas as pd
import pubchempy as pcp
import sys

df = pd.read_csv(sys.argv[1])

mols = df['Name'].tolist()

with open('SDC_dataset_simple.smi', 'w') as smifile:
    for mol in mols:
        canonical_name = mol
        name = mol.replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')
        print(canonical_name+"@"+name)
        data = pcp.get_compounds(mol, 'name')
        line = str(data[0].to_dict(properties=['canonical_smiles'])['canonical_smiles']) + " " + name + '\n'
        smifile.write(line)
        #print(data[0].to_dict(properties=['canonical_smiles'])['canonical_smiles'])
        pcp.download('SDF', name + '.sdf', mol, 'name', overwrite=True)
    # pcp.download('SMILES', name+ '.smi', mol, 'name', overwrite=True)
