# src/models.py
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .utils import save_pickle, load_pickle

BASE_MODEL_NAMES = ["elasticnet", "randomforest", "gbrt"]

def default_models(random_state=42):
    models = {}
    # elastic net pipeline with scaler (linear needs scaling)
    en = Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state))
    ])
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=random_state, n_jobs=-1)
    gbt = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state)
    models["elasticnet"] = en
    models["randomforest"] = rf
    models["gbrt"] = gbt
    return models

def fit_and_save(model, X, y, name):
    model.fit(X, y)
    path = save_pickle(model, f"model_{name}.pkl")
    return path

def load_model(name):
    return load_pickle(f"model_{name}.pkl")
