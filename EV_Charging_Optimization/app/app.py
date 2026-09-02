import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
from datetime import datetime, date, time


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="EV Charging Decision Support",
    page_icon="🔌",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "final_ev_charging_ridge_model.pkl"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# TITLE
# ============================================================

st.title("🔌 EV Charging Demand & Decision Support System")

st.write(
    "Predict charging demand and determine whether additional "
    "charging infrastructure is required."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Station Inputs")


# DATE AND TIME

selected_date = st.sidebar.date_input(
    "Date",
    value=date(2026, 8, 24)
)

selected_time = st.sidebar.time_input(
    "Charging Time",
    value=time(14, 0)
)


# VEHICLE INFORMATION

st.sidebar.subheader("🚗 Vehicle Information")

number_of_vehicles = st.sidebar.number_input(
    "Number of Vehicles",
    min_value=1,
    max_value=500,
    value=50
)

battery_capacity = st.sidebar.slider(
    "Battery Capacity (kWh)",
    20.0,
    150.0,
    60.0,
    5.0
)

initial_soc = st.sidebar.slider(
    "Initial SOC (%)",
    0.0,
    100.0,
    30.0,
    5.0
)

charging_power = st.sidebar.slider(
    "Charging Power (kW)",
    7.0,
    150.0,
    50.0,
    5.0
)


# STATION INFORMATION

st.sidebar.subheader("🔋 Station Information")

existing_chargers = st.sidebar.number_input(
    "Existing Chargers",
    min_value=1,
    max_value=100,
    value=10
)

charger_capacity = st.sidebar.number_input(
    "Charger Capacity (kW)",
    min_value=7.0,
    max_value=350.0,
    value=50.0
)

queue_length = st.sidebar.number_input(
    "Queue Length",
    min_value=0,
    max_value=100,
    value=3
)

station_load = st.sidebar.number_input(
    "Current Station Load (kW)",
    min_value=0.0,
    max_value=5000.0,
    value=250.0
)


# OTHER CONDITIONS

st.sidebar.subheader("🌍 Conditions")

electricity_price = st.sidebar.number_input(
    "Electricity Price (₹/kWh)",
    min_value=1.0,
    max_value=50.0,
    value=8.0
)

renewable_ratio = st.sidebar.slider(
    "Renewable Energy Ratio",
    0.0,
    1.0,
    0.40,
    0.05
)

traffic_density = st.sidebar.selectbox(
    "Traffic Density",
    ["Low", "Medium", "High"]
)

weather_condition = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rainy", "Stormy"]
)

location_type = st.sidebar.selectbox(
    "Location",
    ["Urban", "Suburban", "Highway", "Residential"]
)

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    ["Car", "SUV", "Bus", "Truck", "Motorcycle"]
)

charging_priority = st.sidebar.selectbox(
    "Charging Priority",
    ["Low", "Normal", "High"]
)


# COST

cost_per_charger = st.sidebar.number_input(
    "Cost per Charger (₹)",
    min_value=10000,
    max_value=10000000,
    value=200000,
    step=10000
)


# BUTTON

analyze = st.sidebar.button(
    "🚀 ANALYZE CHARGING STATION",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # DATE TIME
    # --------------------------------------------------------

    dt = datetime.combine(
        selected_date,
        selected_time
    )

    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = dt.weekday()
    day_of_year = dt.timetuple().tm_yday
    week_of_year = dt.isocalendar().week
    quarter = (month - 1) // 3 + 1
    hour = dt.hour
    minute = dt.minute

    is_weekend = int(day_of_week >= 5)
    is_month_start = int(day == 1)

    next_month = (
        dt.replace(day=28) +
        pd.Timedelta(days=4)
    ).replace(day=1)

    last_day = (
        next_month -
        pd.Timedelta(days=1)
    ).day

    is_month_end = int(day == last_day)

    is_peak_hour = int(
        hour in [7, 8, 9, 17, 18, 19, 20]
    )


    # --------------------------------------------------------
    # CYCLICAL FEATURES
    # --------------------------------------------------------

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    day_of_week_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    day_of_week_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )


    # --------------------------------------------------------
    # TIME SLOT
    # --------------------------------------------------------

    if 6 <= hour < 12:
        time_slot = "Morning"

    elif 12 <= hour < 17:
        time_slot = "Afternoon"

    elif 17 <= hour < 22:
        time_slot = "Evening"

    else:
        time_slot = "Night"


    timestamp = dt.strftime(
        "%d-%m-%Y %H:%M"
    )


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "timestamp": timestamp,
        "station_id": "ST001",
        "location_type": location_type,
        "vehicle_type": vehicle_type,
        "arrival_time": timestamp,
        "charging_start_time": timestamp,

        "battery_capacity_kwh": battery_capacity,
        "initial_soc": initial_soc,
        "charging_power_kw": charging_power,
        "queue_length": queue_length,
        "station_load": station_load,
        "electricity_price": electricity_price,
        "renewable_energy_ratio": renewable_ratio,

        "traffic_density": traffic_density,
        "weather_condition": weather_condition,

        "day_of_week": day_of_week,
        "time_slot": time_slot,
        "assigned_charger_id": "CH001",
        "charging_priority": charging_priority,

        "year": year,
        "month": month,
        "day": day,
        "day_of_year": day_of_year,
        "week_of_year": week_of_year,
        "quarter": quarter,
        "hour": hour,
        "minute": minute,

        "is_weekend": is_weekend,
        "is_month_start": is_month_start,
        "is_month_end": is_month_end,
        "is_peak_hour": is_peak_hour,

        "hour_sin": hour_sin,
        "hour_cos": hour_cos,

        "day_of_week_sin": day_of_week_sin,
        "day_of_week_cos": day_of_week_cos,

        "month_sin": month_sin,
        "month_cos": month_cos

    }])


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_demand = float(
        model.predict(input_data)[0]
    )

    predicted_demand = max(
        0,
        predicted_demand
    )


    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    existing_capacity = (
        existing_chargers *
        charger_capacity
    )

    required_chargers = math.ceil(
        predicted_demand /
        charger_capacity
    )

    additional_chargers = max(
        0,
        required_chargers -
        existing_chargers
    )

    capacity_gap = max(
        0,
        predicted_demand -
        existing_capacity
    )


    # --------------------------------------------------------
    # UTILIZATION
    # --------------------------------------------------------

    utilization = (
        predicted_demand /
        existing_capacity
    ) * 100


    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    if utilization <= 70:

        status = "🟢 CAPACITY SUFFICIENT"

        recommendation = (
            "No additional chargers are required."
        )

    elif utilization <= 90:

        status = "🟡 MONITOR DEMAND"

        recommendation = (
            "Current capacity is sufficient, "
            "but demand should be monitored."
        )

    elif utilization <= 100:

        status = "🟠 NEAR CAPACITY"

        recommendation = (
            "The station is close to maximum capacity. "
            "Consider planning additional chargers."
        )

    else:

        status = "🔴 CAPACITY SHORTAGE"

        recommendation = (
            f"Add {additional_chargers} additional "
            "charger(s)."
        )


    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    expansion_cost = (
        additional_chargers *
        cost_per_charger
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("📊 Demand Prediction")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted Demand",
        f"{predicted_demand:.2f} kW"
    )

    col2.metric(
        "Existing Capacity",
        f"{existing_capacity:.2f} kW"
    )

    col3.metric(
        "Utilization",
        f"{utilization:.1f}%"
    )

    col4.metric(
        "Capacity Gap",
        f"{capacity_gap:.2f} kW"
    )


    st.divider()


    # ========================================================
    # CHARGER REQUIREMENT
    # ========================================================

    st.subheader("🔌 Charger Requirement")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Existing Chargers",
        existing_chargers
    )

    col2.metric(
        "Required Chargers",
        required_chargers
    )

    col3.metric(
        "Additional Chargers",
        additional_chargers
    )


    st.divider()


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader("💡 Final Recommendation")

    if utilization > 100:

        st.error(status)

    elif utilization > 90:

        st.warning(status)

    else:

        st.success(status)


    st.info(
        recommendation
    )


    # ========================================================
    # COST
    # ========================================================

    st.subheader("💰 Estimated Expansion Cost")

    st.metric(
        "Estimated Investment",
        f"₹{expansion_cost:,.0f}"
    )


    # ========================================================
    # CHART
    # ========================================================

    st.subheader("📈 Demand vs Capacity")

    chart = pd.DataFrame({

        "Value": [
            predicted_demand,
            existing_capacity
        ]

    }, index=[
        "Predicted Demand",
        "Existing Capacity"
    ])

    st.bar_chart(chart)


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader("📋 Scenario Summary")

    summary = pd.DataFrame({

        "Parameter": [
            "Date",
            "Time",
            "Number of Vehicles",
            "Predicted Demand",
            "Existing Chargers",
            "Charger Capacity",
            "Existing Capacity",
            "Utilization",
            "Required Chargers",
            "Additional Chargers",
            "Capacity Gap",
            "Estimated Investment"
        ],

        "Value": [
            str(selected_date),
            selected_time.strftime("%H:%M"),
            number_of_vehicles,
            f"{predicted_demand:.2f} kW",
            existing_chargers,
            f"{charger_capacity:.0f} kW",
            f"{existing_capacity:.2f} kW",
            f"{utilization:.1f}%",
            required_chargers,
            additional_chargers,
            f"{capacity_gap:.2f} kW",
            f"₹{expansion_cost:,.0f}"
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "👈 Enter the station conditions on the left "
        "and click **🚀 ANALYZE CHARGING STATION**."
    )