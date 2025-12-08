import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from typing import Literal
from FeatureEngineeringFunctions import (
    TransformDateFeature,
    drop_un_needed_features_production,
)
import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))


LOOKUP_DF = None
main_pipeline = None


def load_models(model: Literal["XGBoost", "NN_MLP"]):
    global LOOKUP_DF, main_pipeline, model_path, processor, NN_model, MODEL
    PIPELINE_PATH = ""

    if model == "XGBoost":
        PIPELINE_PATH = "../Models/XGBoost_pipeline.pkl"
        MODEL = "XGB"
    elif model == "NN_MLP":
        model_path = "../Models/best_rossmann_model.keras"
        PIPELINE_PATH = "../Models/NN_preprocessor.pkl"
        MODEL = "NN"

    LOOKUP_PATH = "../Data/preprocessed/full_store_lookup.parquet"

    try:
        print("⏳ Loading Lookup Data...")
        temp_lookup = pd.read_parquet(LOOKUP_PATH)
        temp_lookup = temp_lookup.set_index(["Store", "Date"]).sort_index()
        LOOKUP_DF = temp_lookup
        print("Lookup Loaded.")
    except Exception as e:
        print(f"Error loading Lookup: {e}")
        LOOKUP_DF = None

    try:
        if model == "XGBoost":
            print(f"⏳ Loading {model} Pipeline...")
            main_pipeline = joblib.load(PIPELINE_PATH)
            print("Model XGBoost Components Loaded and Pipeline Assembled.")
        elif model == "NN_MLP":
            NN_model = tf.keras.models.load_model(model_path)
            processor = joblib.load(PIPELINE_PATH)

    except Exception as e:
        print(f"Error loading Pipeline: {e}")
        main_pipeline = None


def preprocess_user_input(input_df):
    processed_df = TransformDateFeature(input_df)
    final_X = drop_un_needed_features_production(processed_df)
    return final_X


def predict_sales(
    # Required
    store_id: int,
    date_str: str,
    # Optional
    scenario_promo: int = None,
    scenario_school_holiday: int = None,
    scenario_distance: float = None,
    scenario_promo2: int = None,
):
    """
    Predicts sales for a specific Store and Date.
    Allows 'What-If' analysis by overriding Promo, Distance, etc.for decision making.
    """

    if LOOKUP_DF is None or (main_pipeline is None and MODEL is None):
        return "Error: Model or Lookup Data not loaded."

    # Fetch Base Data from Lookup (The "Knowledge Base")
    try:
        row_data = LOOKUP_DF.loc[(store_id, date_str)].to_dict()
    except KeyError:
        return f"No data found for Store {store_id} on {date_str}. Date might be out of range or The store was closed."

    # Apply Scenario Logic (User Input vs. Historical )
    final_promo = scenario_promo if scenario_promo is not None else row_data["Promo"]
    final_school = (
        scenario_school_holiday
        if scenario_school_holiday is not None
        else row_data["SchoolHoliday"]
    )
    final_distance = (
        scenario_distance
        if scenario_distance is not None
        else row_data["CompetitionDistance"]
    )
    final_promo2 = (
        scenario_promo2 if scenario_promo2 is not None else row_data["Promo2"]
    )
    # final_day_of_week = (
    #     # scenario_dayOfWeek if scenario_dayOfWeek is not None else
    #     row_data["DayOfWeek"]
    # )

    # 4. Construct the Single-Row DataFrame
    input_df = pd.DataFrame(
        [
            {
                "Store": store_id,
                "Date": pd.to_datetime(date_str),
                # -- Scenario / Defaults --
                "Promo": bool(final_promo),
                "SchoolHoliday": bool(final_school),
                "CompetitionDistance": float(final_distance),
                "Promo2": bool(final_promo2),
                "DayOfWeek": row_data["DayOfWeek"],
                # -- Fixed / Lookup Features --
                "StoreType": row_data["StoreType"],
                "Assortment": row_data["Assortment"],
                "CompetitionDistanceMissing": bool(
                    row_data["CompetitionDistanceMissing"]
                ),
                "CompetitionOpenMissing": bool(row_data["CompetitionOpenMissing"]),
                "OpenDuration": int(row_data["OpenDuration"]),
                "Promo2WeeksDuration": int(row_data["Promo2WeeksDuration"]),
                "IsPromo2Month": bool(row_data["IsPromo2Month"]),
                "DaysUntilNextStateHoliday": int(row_data["DaysUntilNextStateHoliday"]),
                "DaysSinceLastStateHoliday": int(row_data["DaysSinceLastStateHoliday"]),
                "DaysUntilClosed": int(row_data["DaysUntilClosed"]),
            }
        ]
    )

    # 5. Preprocess and Predict
    final_X = preprocess_user_input(input_df)

    if MODEL == "XGB":
        log_pred = main_pipeline.predict(final_X)
        prediction = np.expm1(log_pred)[0]
    elif MODEL == "NN":
        data_transformed = processor.transform(final_X)
        prediction_logged = NN_model.predict(data_transformed)
        prediction = np.expm1(prediction_logged)[0][0]

    return max(prediction, 0)
