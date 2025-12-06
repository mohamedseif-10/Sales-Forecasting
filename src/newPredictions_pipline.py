import pandas as pd
import numpy as np
import joblib
from typing import Literal
from sklearn.pipeline import Pipeline
from FeatureEngineeringFunctions import TransformDateFeature, drop_un_needed_features_production

# def loadData(mode: Literal["train", "test"]) -> pd.DataFrame:

def load_models(model: Literal["XGBoost", "LSTM"]):
    if model == "XGBoost":
        PIPELINE_PATH ="../Models/XGBoost_pipeline.pkl"
    elif model == "LSTM":
        PIPELINE_PATH = 'lstm'

    LOOKUP_PATH = "../Data/preprocessed/full_store_lookup.parquet"

    try:
        print("⏳ Loading Lookup Data...")
        LOOKUP_DF = pd.read_parquet(LOOKUP_PATH)
        LOOKUP_DF = LOOKUP_DF.set_index(["Store", "Date"]).sort_index()
        print("Lookup Loaded.✅")
    except Exception as e:
        print(f"Error loading Lookup: {e}")
        LOOKUP_DF = None

    try:
        main_pipeline = joblib.load(PIPELINE_PATH)
        print("✅Model Components Loaded and Pipeline Assembled.✅")
    except Exception as e:
        print(f"Error loading Pipeline: {e}")
        main_pipeline = None


def preprocess_user_input(input_df):
    processed_df = TransformDateFeature(input_df)
    final_X = drop_un_needed_features_production(processed_df)
    # final_X = processed_df.drop(columns=["date", "DayOfWeek"], axis=1)

    final_X.info()
    return final_X
