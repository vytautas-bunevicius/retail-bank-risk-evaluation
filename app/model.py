import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint["model"], checkpoint["selected_features"]

def predict(model, df):
    return model.predict_proba(df)[0, 1]
