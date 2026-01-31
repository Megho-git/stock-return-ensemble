# src/stack.py
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from src.utils import save_pickle

def train_meta_model(
    base_models: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """
    Train meta model on validation predictions from base models.
    """
    preds = []

    for name, model in base_models.items():
        preds.append(model.predict(X_val))

    Z = np.column_stack(preds)

    meta = ElasticNetCV(
        cv=5,
        l1_ratio=[0.2, 0.5, 0.8],
        max_iter=5000,
        random_state=42,
    )

    meta.fit(Z, y_val.values)

    save_pickle(meta, "meta_elasticnet.pkl")
    return meta
