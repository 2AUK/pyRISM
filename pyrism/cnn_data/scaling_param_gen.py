import numpy as np
from pathlib import Path
import pandas as pd
import numpy as np

files = Path('./model').resolve()
df_data_1 = pd.read_csv("multi-T_gf_dataset_water-both_chloroform_carbontet.csv")
df_data_2 = pd.read_csv("free_energy_neutral_ionised_dataset_gf.csv")
df_data = pd.concat([df_data_1, df_data_2], ignore_index=True).drop_duplicates()
for f in files.iterdir():
    means = []
    stds = []
    print(f.stem)
    df = pd.read_csv(str(f) + '/' + f.stem + '/' + f.stem + '_train_plot.csv').sort_values(by=['Mol', 'Temp'], ignore_index=True)
    df_train = pd.merge(df_data.sort_values(by='Temp'), df.sort_values(by='Temp'), how='inner', on=['Mol', 'Temp']).sort_values(by=['Mol', 'Temp'], ignore_index=True)
    assert(len(df_train.index) == len(df.index))
    df_train = df_train.drop(columns='y_pred')
    for column in df_train.columns[3:]:
        means.append(df_train[column].mean())
        stds.append(df_train[column].std())

    data = np.asarray([means, stds])
    columns = df_train.columns[3:].values.tolist()
    df_scaling_params = pd.DataFrame(data, columns=columns)
    df_scaling_params.to_csv(str(f) + '/' + f.stem + '/' + f.stem + '_scaling_params.csv', index=False)