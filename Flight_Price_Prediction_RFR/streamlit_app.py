import streamlit as st
import joblib
import numpy as np

# -------------------------------------------------
# Page Config 
# -------------------------------------------------
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "flight_price_xgboost.pkl")
    return joblib.load(model_path)

model = load_model()

# -------------------------------------------------
# Custom Styling
# -------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 25px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        color: #0b5394;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="main-title">✈️ Flight Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict airline ticket prices using Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🧾 Flight Details")

airline_map = {
    "Air India": 1,
    "IndiGo": 3,
    "Jet Airways": 4,
    "SpiceJet": 6,
    "Vistara": 7
}

source_map = {
    "Banglore": 0,
    "Chennai": 1,
    "Delhi": 2,
    "Kolkata": 3,
    "Mumbai": 4
}

destination_map = {
    "Banglore": 0,
    "Cochin": 1,
    "Delhi": 2,
    "Hyderabad": 3,
    "Kolkata": 4,
    "New Delhi": 5
}

airline = st.sidebar.selectbox("✈️ Airline", airline_map.keys())
source = st.sidebar.selectbox("📍 Source City", source_map.keys())
destination = st.sidebar.selectbox("🏁 Destination City", destination_map.keys())

total_stops = st.sidebar.selectbox(
    "⏱ Total Stops",
    ["Non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
)

# -------------------------------------------------
# Main Inputs
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    day = st.slider("📅 Journey Day", 1, 31, 15)
    month = st.slider("📅 Journey Month", 1, 12, 6)
    year = st.number_input("📅 Year", value=2019)

with col2:
    dep_hour = st.slider("🕒 Departure Hour", 0, 23, 10)
    dep_min = st.slider("🕒 Departure Minute", 0, 59, 30)

with col3:
    arr_hour = st.slider("🕓 Arrival Hour", 0, 23, 12)
    arr_min = st.slider("🕓 Arrival Minute", 0, 59, 30)

st.markdown("### ⌛ Flight Duration")

col4, col5 = st.columns(2)
with col4:
    dur_hour = st.slider("Duration (Hours)", 0, 30, 2)
with col5:
    dur_min = st.slider("Duration (Minutes)", 0, 59, 30)

# -------------------------------------------------
# Auto Route Encoding (clean UI)
# -------------------------------------------------
def auto_route_encoding(stops):
    route = [0, 0, 0, 0, 5]
    if stops == "Non-stop":
        route[0] = 0
    elif stops == "1 stop":
        route[1] = 10
    elif stops == "2 stops":
        route[2] = 20
    elif stops == "3 stops":
        route[3] = 30
    return route

stops_map = {
    "Non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
}

route_1, route_2, route_3, route_4, route_5 = auto_route_encoding(total_stops)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.markdown("---")

if st.button("🔮 Predict Flight Price", use_container_width=True):

    input_data = np.array([[
        airline_map[airline],          # Airline
        source_map[source],            # Source
        destination_map[destination],  # Destination
        stops_map[total_stops],        # Total_Stops
        day,                   # Journey_Day
        month,                 # Journey_Month
        dep_hour,                      # Dep_Hour
        dep_min,                       # Dep_Min
        arr_hour,                      # Arrival_Hour
        arr_min,                       # Arrival_Min
        dur_hour,                      # Duration_hours
        dur_min                        # Duration_mins
    ]])

    # ✅ Predict (log → original scale)
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)

    st.markdown(
        f'<div class="result-box">💰 Estimated Price: ₹ {int(prediction)}</div>',
        unsafe_allow_html=True
    )
# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("📊 Machine Learning Project | Streamlit Deployment")
