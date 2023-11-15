import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from pathlib import Path
from mordred import Calculator, VdwVolumeABC, McGowanVolume
from rdkit import Chem
from rdkit.Chem import AllChem

ratkova_dimensionless_pmv = { "methane": 1.79, "ethane": 2.45, "propane": 3.00, "butane": 3.47, "pentane": 4.04, "hexane": 4.54, "heptane": 5.04, "octane": 5.57, "nonane": 6.05, "decane": 6.57 }

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

class Molecule:
    def __init__(self, fname, density):
        self.name = fname.stem
        self.ratkova_pmv = ratkova_dimensionless_pmv[fname.stem]
        self.pyrism_pmv = self.read_td(fname, density)
        self.smi = self.read_smi(fname)
        self.vdw_volume = self.compute_vdw_abc_volume(self.smi, density)
        self.mol_volume = self.compute_mol_volume(self.smi, density)
        self.mcgowan_volume = self.compute_mcgowan_volume(self.smi, density)

    def read_td(self, fname, density):
        with open(fname, 'r') as tdfile:
            for line in tdfile:
                line = line.split()
                if line[0] == "RISM":
                    return float(line[-2]) * density

    def compute_vdw_abc_volume(self, smi, density):
        mol = Chem.MolFromSmiles(smi)
        result = Calculator(VdwVolumeABC.VdwVolumeABC)(mol)
        return result.ix[0] * 1e24 / 6.022e23 * density

    def compute_mcgowan_volume(self, smi, density):
        mol = Chem.MolFromSmiles(smi)
        result = Calculator(McGowanVolume.McGowanVolume)(mol)
        return result.ix[0] * density * 1e24 / 6.022e23

    def compute_mol_volume(self, smi, density):
        mol = Chem.MolFromSmiles(smi)
        AllChem.EmbedMolecule(mol)
        result = AllChem.ComputeMolVolume(mol)
        return result * density 

    def read_smi(self, fname):
        smi_path = Path("linear_alkanes") / Path(fname.stem + ".smi")
        with open(smi_path, 'r') as smifile:
            return smifile.readline().split('\n')[0]

    def __str__(self):
        return "Name: {}\nRatkova PMV: {}\npyRISM PMV: {}\nDifference: {}\nSMILES: {}\nvdW ABC Volume: {}\nMcGowan Volume: {}\nMolecular Volume (RDKit): {}".format(self.name, self.ratkova_pmv, self.pyrism_pmv, self.ratkova_pmv - self.pyrism_pmv, self.smi, self.vdw_volume, self.mcgowan_volume, self.mol_volume)


class Plotter():
    def __init__(self, molecules):
        self.pyrism_pmvs = []
        self.ratkova_pmvs = []
        self.mol_volumes = []
        self.mcgowan_volumes = []
        self.vdw_volumes = []

        for mol in molecules:
            self.pyrism_pmvs.append(mol.pyrism_pmv)
            self.ratkova_pmvs.append(mol.ratkova_pmv)
            self.mol_volumes.append(mol.mol_volume)
            self.mcgowan_volumes.append(mol.mcgowan_volume)
            self.vdw_volumes.append(mol.vdw_volume)
    
        regression_results(self.pyrism_pmvs, self.ratkova_pmvs)    
        self.plot(self.pyrism_pmvs, self.ratkova_pmvs, "pyRISM vs Ratkova (RISM-MOL)", "pyRISM", "Ratkova (RISM-MOL)")
        regression_results(self.pyrism_pmvs, self.mol_volumes)    
        self.plot(self.pyrism_pmvs, self.mol_volumes, "pyRISM vs Molecular Volume (RDkit)", "pyRISM", "Molecular Volume (RDkit)")
        regression_results(self.pyrism_pmvs, self.mcgowan_volumes)    
        self.plot(self.pyrism_pmvs, self.mcgowan_volumes, "pyRISM vs McGowan Volume (Mordred)", "pyRISM", "McGowan Volume (Mordred)")
        regression_results(self.pyrism_pmvs, self.vdw_volumes)    
        self.plot(self.pyrism_pmvs, self.vdw_volumes, "pyRISM vs vdW ABC Volume (Mordred)", "pyRISM", "vdW ABC Volume (Mordred)")

    def plot(self, x, y, title, xlabel, ylabel):
        a, b = np.polyfit(x, y, 1)
        _, ax = plt.subplots()
        ax.axline((0,0), (2.5, 2.5), color='k', linestyle='--')
        ax.axline((x[0], y[0]), slope=a, color='orange')
        plt.plot(x, y, 'r.')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        r2 = round(metrics.r2_score(x, y), 4)
        rmse = round(np.sqrt(metrics.mean_squared_error(x, y)), 4)

        plt.text(0.5, 4, 'y = {:.2f}x + {:.2f}\nR^2: {} RMSE: {}'.format(a, b, r2, rmse), size=10)
        plt.show()

p = Path('.')
inputs = list(p.glob("*.td"))
molecules = []
density = 0.0337
for i in inputs:
    molecules.append(Molecule(i, density))

Plotter(molecules)
#
# for i in inputs:
#     with open(i, 'r') as tdfile:
#         for line in tdfile:
#             line = line.split()
#             if line[0] == "RISM":
#                 pyrism_dimensionless_pmv.append(float(line[-2]))
#                 
# pyrism_dimensionless_pmv = np.asarray(sorted(pyrism_dimensionless_pmv))
# pyrism_dimensionless_pmv *= density
# a, b = np.polyfit(ratkova_dimensionless_pmv, pyrism_dimensionless_pmv, 1)
# _, ax = plt.subplots()
# ax.axline((0.0, 0.0), (2.5, 2.5), color='k', linestyle='--')
# ax.axline((ratkova_dimensionless_pmv[0], pyrism_dimensionless_pmv[0]), slope=a, color='orange')
# plt.plot(ratkova_dimensionless_pmv, pyrism_dimensionless_pmv, 'r.')
# plt.xlabel("Ratkova PMV")
# plt.ylabel("pyRISM PMV")
# plt.title("Dimensionless PMV Correlation")
# plt.text(1, 6, 'y = {:.2f}x + {:.2f}'.format(a, b), size=10)
# plt.show()
#
# smiles = []
# for i in inp_smi:
#     with open(i, 'r') as smifile:
#         smiles.append(smifile.readline().split('\n')[0])
#
# print(smiles)
