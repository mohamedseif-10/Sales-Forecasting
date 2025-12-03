import pandas as pd
import numpy as np


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


def loadAndPrepareTest(test_path: str):
    """Cleans and preprocesses the unseen test data."""
    
    test_data_types = {
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
        "PromoInterval": "category"}
        
    test_data = pd.read_csv(
        test_path,
        parse_dates=["Date"],
        encoding="utf-8",
        low_memory=False,
        dtype=test_data_types,)
    
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
    test_data.drop(columns=["Open"], axis=1, inplace=True)
    test_data = TransformDateFeature(test_data)
    test_data = feature_competition_duration(test_data)
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
    data.drop(columns=["OpenDate", 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Date'], axis=1, inplace=True)
    
    return data