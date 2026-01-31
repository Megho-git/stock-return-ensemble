# src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def metrics_df(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sign_acc = (np.sign(y_true) == np.sign(y_pred)).mean()
    return {"MAE": mae, "RMSE": rmse, "SignAcc": sign_acc}

def save_predictions_csv(df_preds, path):
    df_preds.to_csv(path, index=False)
