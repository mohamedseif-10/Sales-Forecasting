import streamlit as st
import pandas as pd
from datetime import date
from FullPipelineForBothModels import predict_sales, load_models
import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))


st.set_page_config(page_title="Sales Forecasting Using NN", page_icon="📈")


@st.cache_resource
def init_application(model_name: str = "NN_MLP"):
    """
    Loads models based on the selected model_name.
    Caching ensures this runs only once per session/model type.
    """
    try:
        load_models(model_name)
        return True
    except Exception as e:
        return str(e)


# Initialize immediately
model_choice = "NN_MLP"
status = init_application(model_choice)

if status != True:
    st.error(f"Failed to load models: {status}")
    st.stop()

st.title("Sales Forecasting 📈")
# st.info(f"Using Model: {model_choice}")


# model_choice = st.selectbox(
#     "Select Predictive Model",
#     options=["XGBoost", "NN_MLP"],
#     index=0,
#     help="Switch between the XGBoost and NN_MLP pipelines.",
# )

try:
    with st.spinner(f"Loading {model_choice} pipeline..."):
        if status == True:
            st.success(f"Using model: **{model_choice}**")
        else:
            st.error(f"error loading the model")
except Exception as e:
    st.error(f"Error loading model {model_choice}: {e}. Please check file paths.")


store_id = st.number_input("Store ID", min_value=1, max_value=1115, step=1, value=1)

st.subheader("Select Date")
col1, col2, col3 = st.columns(3)
with col1:
    year = st.number_input("Year", min_value=2013, max_value=2015, step=1, value=2015)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, step=1, value=7)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, step=1, value=31)

valid_date = None
date_is_valid = True

try:
    valid_date = date(int(year), int(month), int(day))
    # Validate full date range based on our dataset range
    min_date = date(2013, 1, 1)
    max_date = date(2015, 9, 17)

    if not (min_date <= valid_date <= max_date):
        st.error(f"Date must be between {min_date} and {max_date}")
        date_is_valid = False
except ValueError:
    st.error("Invalid date combination (check day/month/year)")
    date_is_valid = False

st.markdown("---")
st.subheader("Scenarios for (What-If Analysis)")
st.caption(
    "Leave it as 'Default' to use the actual values in real for selected Date/Store"
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    promo_input = st.selectbox(
        "Promo Active?",
        options=["Default", "Yes", "No"],
        index=0,
        key="promo_key",
    )

if promo_input == "Default":
    promo = None
elif promo_input == "Yes":
    promo = True
else:
    promo = False

with col_b:
    school_holiday_input = st.selectbox(
        "School Holiday?",
        options=["Default", "Yes", "No"],
        index=0,
        key="holiday_key",
    )

if school_holiday_input == "Default":
    school_holiday = None
elif school_holiday_input == "Yes":
    school_holiday = True
else:
    school_holiday = False

with col_c:
    promo2_input = st.selectbox(
        "Promo2 Active?",
        options=["Default", "Yes", "No"],
        index=0,
        key="promo2_key",
    )

if promo2_input == "Default":
    promo2 = None
elif promo2_input == "Yes":
    promo2 = True
else:
    promo2 = False

competition_distance = st.number_input(
    "Competition Distance (meters)",
    min_value=0.0,
    step=10.0,
    value=None,
    format="%.2f",
)


if st.button("Predict Sales", type="primary"):
    if date_is_valid:

        optional_params = {}

        if promo is not None:
            optional_params["scenario_promo"] = int(promo)

        if school_holiday is not None:
            optional_params["scenario_school_holiday"] = int(school_holiday)

        if promo2 is not None:
            optional_params["scenario_promo2"] = int(promo2)

        if competition_distance is not None:
            optional_params["scenario_distance"] = float(competition_distance)

        date_str_formatted = valid_date.strftime("%Y-%m-%d")
        with st.spinner(f"Calculating prediction using {model_choice}..."):
            result = predict_sales(
                store_id=int(store_id),
                date_str=date_str_formatted,
                # scenario_promo=int(promo),
                # scenario_school_holiday=int(school_holiday),
                # scenario_distance=float(competition_distance),
                # scenario_promo2=int(promo2),
                # Optional Keyword Arguments (only non-None values are passed)
                **optional_params,
            )

        if isinstance(result, str) and "Error" in result:
            st.error(result)
        else:
            st.success("Prediction Successful!")

            st.metric(
                label=f"Predicted Sales for Store {store_id})",
                value=f"${result:,.2f}",
            )

    else:
        st.warning("Please fix the date errors above before submitting.")
