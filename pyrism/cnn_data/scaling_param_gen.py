import numpy as np
from pathlib import Path
import pandas as pd

files = Path('./model').resolve()
df_data_1 = pd.read_csv("multi-T_gf_dataset_water-both_chloroform_carbontet.csv")
df_data_2 = pd.read_csv("free_energy_neutral_ionised_dataset_gf.csv")
df_data = pd.concat([df_data_1, df_data_2], ignore_index=True).drop_duplicates()
for f in files.iterdir():
    print(f.stem)
    df = pd.read_csv(str(f) + '/' + f.stem + '/' + f.stem + '_train_plot.csv')
    df_train = pd.merge(df_data.sort_values(by='Temp'), df.sort_values(by='Temp'), how='inner', on=['Mol', 'Temp'])
    print(df_train.sort_values(by='Mol'))
