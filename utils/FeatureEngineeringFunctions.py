import pandas as pd
import numpy as np
from typing import Literal


def TransformDateFeature(data):
    """Extracts date features from the 'Date' column in the DataFrame."""
    data["Year"] = data["Date"].dt.year.astype(np.int16)
    data["Month"] = data["Date"].dt.month.astype(np.int8)
    data["Day"] = data["Date"].dt.day.astype(np.int8)

    data["DayOfYear"] = data["Date"].dt.dayofyear.astype(np.int16)
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(np.int8)
    data["IsLastDayOfMonth"] = data["Date"].dt.is_month_end.astype(bool)

    # Cyclical features
    data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7).astype(np.float32)
    data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7).astype(np.float32)
    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12).astype(np.float32)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12).astype(np.float32)

    data["IsWeekend"] = data["DayOfWeek"].isin([6, 7]).astype(bool)
    data["IsMonthEnd"] = data["Date"].dt.is_month_end.astype(bool)
    data["IsMonthStart"] = data["Date"].dt.is_month_start.astype(bool)
    return data


def loadData(mode: Literal["train", "test"]) -> pd.DataFrame:
    """
    Optimization for two separated load function in only one general function with param
    Args:
        path (str): The file path for the data
        mode (Literal['train', 'test']): Specifies which dataset to load.

    Returns:
        pd.DataFrame: The loaded and typed DataFrame.
    """

    data_types_train = {
        "Store": "int16",
        "DayOfWeek": "int8",
        "CompetitionDistance": "float32",
        "CompetitionOpenSinceMonth": "float32",
        "CompetitionOpenSinceYear": "float32",
        "CompetitionDistanceMissing": "bool",
        "CompetitionOpenMissing": "bool",
        "StateHoliday": "category",
        "SchoolHoliday": "int8",
        "Promo": "bool",
        "Promo2": "bool",
        "Promo2SinceYear": "float32",
        "Promo2SinceWeek": "float32",
        "PromoInterval": "category",
        "StoreType": "category",
        "Assortment": "category",
        "Sales": "float32",
    }

    data_types_test = {
        "Store": "int16",
        "DayOfWeek": "int8",
        "Open": "float32",
        "Promo": "bool",
        "SchoolHoliday": "int8",
        "StateHoliday": "category",
        "StoreType": "category",
        "Assortment": "category",
        "CompetitionDistance": "float32",
        "CompetitionOpenSinceMonth": "float32",
        "CompetitionOpenSinceYear": "float32",
        "Promo2": "bool",
        "Promo2SinceWeek": "float32",
        "Promo2SinceYear": "float32",
        "PromoInterval": "category",
    }

    file_path = "../Data/intermediate/"
    if mode == "train":
        file_path = file_path + "cleaned_training_data.csv"
        dtypes = data_types_train
        kwargs = {"index_col": "Date_index"}
    elif mode == "test":
        file_path = file_path + "merged_testing_data.csv"
        dtypes = data_types_test
        kwargs = {}
    else:
        raise ValueError("Mode must be 'train' or 'test'.")

    data = pd.read_csv(
        file_path,
        dtype=dtypes,
        parse_dates=["Date"],
        encoding="utf-8",
        low_memory=False,
        **kwargs
    )

    return data


# def loadTest(test_path: str):
#     """Cleans and preprocesses the unseen test data."""

#     test_data_types = {
#         "Store": "int16",
#         "DayOfWeek": "int8",
#         "Open": "float32",
#         "Promo": "bool",
#         "SchoolHoliday": "int8",
#         "StateHoliday": "category",
#         "StoreType": "category",
#         "Assortment": "category",
#         "CompetitionDistance": "float32",
#         "CompetitionOpenSinceMonth": "float32",
#         "CompetitionOpenSinceYear": "float32",
#         "Promo2": "bool",
#         "Promo2SinceWeek": "float32",
#         "Promo2SinceYear": "float32",
#         "PromoInterval": "category",
#     }


#     test_data = pd.read_csv(
#         test_path,
#         parse_dates=["Date"],
#         encoding="utf-8",
#         low_memory=False,
#         dtype=test_data_types
#     )
#     return test_data

# def loadTrain(train_path: str):
#     data_types = {
#         "Store": "int16",
#         "DayOfWeek": "int8",
#         "CompetitionDistance": "float32",
#         "CompetitionOpenSinceMonth": "float32",
#         "CompetitionOpenSinceYear": "float32",
#         "CompetitionDistanceMissing": "bool",
#         "CompetitionOpenMissing": "bool",
#         "StateHoliday": "category",
#         "SchoolHoliday": "int8",
#         "Promo": "bool",
#         "Promo2": "bool",
#         "Promo2SinceYear": "float32",
#         "Promo2SinceWeek": "float32",
#         "PromoInterval": "category",
#         "StoreType": "category",
#         "Assortment": "category",
#         "Sales": "float32",
#     }

#     data = pd.read_csv(
#         "../Data/intermediate/cleaned_training_data.csv",
#         dtype=data_types,
#         parse_dates=["Date"],
#         index_col="Date_index",
#         encoding="utf-8",
#     )

#     return data


def initial_cleaning(test_data):
    test_data["PromoInterval"] = test_data["PromoInterval"].cat.add_categories("no_promo")
    test_data["PromoInterval"] = test_data["PromoInterval"].fillna("no_promo")
    test_data["Promo2SinceWeek"] = test_data["Promo2SinceWeek"].fillna(0)
    test_data["Promo2SinceYear"] = test_data["Promo2SinceYear"].fillna(0)
    test_data["CompetitionDistanceMissing"] = np.where(test_data["CompetitionDistance"].isna(), 1, 0)
    test_data["CompetitionOpenMissing"] = np.where(
        test_data["CompetitionOpenSinceMonth"].isna()| test_data["CompetitionOpenSinceYear"].isna(),1,0)
    test_data["CompetitionOpenSinceYear"] = test_data["CompetitionOpenSinceYear"].fillna(1995)
    test_data["CompetitionOpenSinceMonth"] = test_data["CompetitionOpenSinceMonth"].fillna(1)
    test_data["CompetitionDistanceMissing"] = test_data["CompetitionDistanceMissing"].astype("bool")
    test_data["CompetitionOpenMissing"] = test_data["CompetitionOpenMissing"].astype("bool")
    return test_data

def feature_competition_duration(data):
    """Creates 'OpenDuration' feature based on competition open date and current date."""
    data["OpenDate"] = pd.to_datetime(
        data["CompetitionOpenSinceYear"].astype(int).astype(str)
        + "-"
        + data["CompetitionOpenSinceMonth"].astype(int).astype(str)
        + "-01",
        errors="coerce")
    
    data["OpenDuration"] = (
    (data["Date"].dt.year - data["OpenDate"].dt.year) * 12
    + (data["Date"].dt.month - data["OpenDate"].dt.month)).astype(np.int16)
    
    data["OpenDuration"] = data["OpenDuration"].apply(
        lambda x: 24 if x > 24 else (0 if x < 0 else x)).astype(np.int8)
    return data

def feature_engineer_promo2(data):
    """
    Creates Promo2 new features: duration, start seasonality,
    and the "Is Active" flag, relying on numerical math instead of date conversion.
    """

    data["Promo2WeeksDuration"] = (data["Year"] - data["Promo2SinceYear"]) * 52 + (
        data["WeekOfYear"] - data["Promo2SinceWeek"])
    data["Promo2WeeksDuration"] = data["Promo2WeeksDuration"].clip(lower=0, upper=25).astype(np.int8)

    data.loc[data["Promo2"] == 0, "Promo2WeeksDuration"] = 0
    data["Promo2SinceMonth"] = (np.ceil(data["Promo2SinceWeek"] / 4.0)).astype(np.int8)

    # data.loc[data["Promo2"] == 0, "Promo2SinceMonth"] = 0

    month_to_str = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    data["MonthStr"] = data["Month"].map(month_to_str)

    def check_promo_active(row): 
        if row["Promo2"] == 1 and row["MonthStr"] in row["PromoInterval"]:
            return 1
        return 0

    data["IsPromo2Month"] = data.apply(check_promo_active, axis=1).astype(np.bool)
    return data

def feature_engineer_counters(data):
    """
    Generates time-counters with vectorized approach 100Times faster
    1. DaysUntilStateHoliday (Captures rare, massive events)
    2. DaysUntilClosed (Captures frequent weekly cycle + holidays)
    """

    data = data.sort_values(["Store", "Date"])
    data["StateHoliday"] = data["StateHoliday"].astype(str)
    is_holiday = data["StateHoliday"] != "0"
    data["HolidayDate"] = np.where(is_holiday, data["Date"], pd.NaT)
    data["HolidayDate"] = pd.to_datetime(data["HolidayDate"])

    next_holiday = data.groupby("Store")["HolidayDate"].bfill()
    data["DaysUntilNextStateHoliday"] = (next_holiday - data["Date"]).dt.days
    data["DaysUntilNextStateHoliday"] = data["DaysUntilNextStateHoliday"].fillna(99)
    data["DaysUntilNextStateHoliday"] = (data["DaysUntilNextStateHoliday"].clip(upper=14).astype(np.int8))

    last_holiday = data.groupby("Store")["HolidayDate"].ffill()
    data["DaysSinceLastStateHoliday"] = (data["Date"] - last_holiday).dt.days
    data["DaysSinceLastStateHoliday"] = data["DaysSinceLastStateHoliday"].fillna(99)
    data["DaysSinceLastStateHoliday"] = (data["DaysSinceLastStateHoliday"].clip(upper=14).astype(np.int8))

    mask_closed = (data["DayOfWeek"] == 7) | (data["StateHoliday"] != "0")
    data["ClosureDate"] = np.where(mask_closed, data["Date"], pd.NaT)
    data["ClosureDate"] = pd.to_datetime(data["ClosureDate"])

    next_closure = data.groupby("Store")["ClosureDate"].bfill()
    data["DaysUntilClosed"] = (next_closure - data["Date"]).dt.days

    data["DaysUntilClosed"] = (data["DaysUntilClosed"].fillna(7).clip(upper=7).astype(np.int8))
    data["StateHoliday"] = data["StateHoliday"].astype(str).astype("category")
    return data

def drop_un_needed_features(data):
    columns_to_drop = [
        "Open",
        "Date",
        "HolidayDate",
        "ClosureDate",
        "OpenDate",
        "CompetitionOpenSinceYear",
        "CompetitionOpenSinceMonth",
        "Promo2SinceYear",
        "Promo2SinceWeek",
        "PromoInterval",
        "MonthStr",
        "DayOfWeek", 
        "StateHoliday",
        "Promo2SinceMonth",
    ]
    data = data.drop(columns=columns_to_drop, axis=1)
    return data

    # test_data.drop(columns=["Open", "Date"], axis=1, inplace=True)
    # data.drop(columns=["OpenDate", 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], axis=1, inplace=True)
    # data.drop(["Promo2SinceYear", "Promo2SinceWeek", "PromoInterval", "MonthStr"],axis=1,inplace=True)
    # data.drop(["HolidayDate", "ClosureDate", "Date"], axis=1, inplace=True)

def PrepareTest():
    """
    Combined feature engineering pipeline!
    """
    test_data = loadData('test')
    test_data = initial_cleaning(test_data)
    test_data = TransformDateFeature(test_data)
    test_data = feature_competition_duration(test_data)
    test_data = feature_engineer_promo2(test_data)
    test_data = feature_engineer_counters(test_data)
    test_data = drop_un_needed_features(test_data)
    return test_data
