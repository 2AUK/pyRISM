import numpy as np
from pathlib import Path
import pandas as pd

files = Path('./model').resolve()

for f in files.iterdir():
    print(f.stem)
    df = pd.read_csv(str(f) + '/' + f.stem + '/' + f.stem + '_train_plot.csv')
    y = df['y'].to_numpy()
    y_pred = df['y_pred'].to_numpy()
    print(y.mean(), y.std())
    print(y_pred.mean(), y_pred.std())