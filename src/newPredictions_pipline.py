import pandas as pd
import numpy as np
import joblib
from typing import Literal
from FeatureEngineeringFunctions import (TransformDateFeature,drop_un_needed_features_production,)
import os
import sys
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))


LOOKUP_DF = None
main_pipeline = None


def load_models(model: Literal["XGBoost", "LSTM"]):
    global LOOKUP_DF, main_pipeline
    PIPELINE_PATH = ""

    if model == "XGBoost":
        PIPELINE_PATH = "../Models/XGBoost_pipeline.pkl"
    elif model == "LSTM":
        PIPELINE_PATH = "../Models/LSTM.pkl"

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
        print(f"⏳ Loading {model} Pipeline...")
        main_pipeline = joblib.load(PIPELINE_PATH)
        print("Model Components Loaded and Pipeline Assembled.")
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
    scenario_dayOfWeek: int = None,
):
    """
    Predicts sales for a specific Store and Date.
    Allows 'What-If' analysis by overriding Promo, Distance, etc.for decision making.
    """

    if LOOKUP_DF is None or main_pipeline is None:
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
    final_day_of_week = (
        scenario_dayOfWeek if scenario_dayOfWeek is not None else row_data["DayOfWeek"]
    )

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
                "DayOfWeek": int(final_day_of_week),
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

    log_pred = main_pipeline.predict(final_X)
    prediction = np.expm1(log_pred)[0]

    return max(prediction, 0)


# if __name__ == "__main__":
#     load_models("XGBoost")

#     if main_pipeline is not None and LOOKUP_DF is not None:
#         print("\n--- Test 1: Historical Prediction (Train Data) ---")
#         sales_a = predict_sales(store_id=1, date_str="2015-07-31")
#         print(f"Store 1 on 2014-05-05 Sales Prediction: ${sales_a:,.2f}")
#     else:
#         print("Skipping prediction because loading failed.")
