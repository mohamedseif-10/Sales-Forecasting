import pandas as pd
import numpy as np


def TransformDateFeature(data):
    """Extracts date features from the 'Date' column in the DataFrame."""
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day

    data["DayOfYear"] = data["Date"].dt.dayofyear
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

    # Sales often spike at the beginning and end of the month.
    # This flag captures the last few days of the month.
    data["IsLastDayOfMonth"] = data["Date"].dt.is_month_end.astype(bool)
    import numpy as np

    # 1. Cyclical Encoding for DayOfWeek -- it's important to capture the cyclical nature of days in a week
    # NN may misinterpret Monday (1) and Sunday (7) as being far apart numerically the same as 1-12 for months
    data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7).astype(np.float32)
    data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7).astype(np.float32)

    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12).astype(np.float32)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12).astype(np.float32)

    data["IsWeekend"] = data["DayOfWeek"].isin([6, 7]).astype(bool)
    data["IsMonthEnd"] = data["Date"].dt.is_month_end.astype(bool)
    data["IsMonthStart"] = data["Date"].dt.is_month_start.astype(bool)

    data.drop("Date", axis=1, inplace=True)
    return data


def clean_unseen_test_data(test_path: str):
    test_data_types = {
        "Store": "int16",
        "DayOfWeek": "int8",
        "Open": "float32",
        "Promo": "int8",
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
    return test_data

test_data = clean_unseen_test_data("../Data/Preprocessed_data/merged_data_before_preprocessing/merged_testing_data.csv")
