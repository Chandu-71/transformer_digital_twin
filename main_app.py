import streamlit as st
from feature1_predictive_balancer import render_realtime_grid
from feature2_capacity_planner import render_feature2

# ------------------ PAGE CONFIG ------------------
# MUST be the first Streamlit command!
st.set_page_config(
    page_title="NITW Smart Grid Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ STATE INIT & CALLBACKS ------------------
if "temperature" not in st.session_state:
    st.session_state.temperature = 72
if "load1" not in st.session_state:
    st.session_state.load1 = 65
if "load2" not in st.session_state:
    st.session_state.load2 = 40
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = "Normal"
if "current_routing" not in st.session_state:
    st.session_state.current_routing = "NORMAL"
if "expected_drop" not in st.session_state:
    st.session_state.expected_drop = 0
if "applied_routing" not in st.session_state:
    st.session_state.applied_routing = "NORMAL"
if "action_applied_once" not in st.session_state:
    st.session_state.action_applied_once = False

# --- PROCESS AI ACTION BEFORE WIDGETS RENDER ---
# --- PROCESS AI ACTION BEFORE WIDGETS RENDER ---
if st.session_state.get("apply_action") and not st.session_state.get("action_applied_once"):
    routing = st.session_state.current_routing
    drop = st.session_state.expected_drop

    # Simulating partial load sharing (Notice how this is indented INWARD)
    if routing == "TIER_1_L2":
        st.session_state.load2 = int(st.session_state.load2 * 0.5)
    elif routing == "TIER_1_L1":
        st.session_state.load1 = int(st.session_state.load1 * 0.5)
    elif routing == "TIER_2_L2":          
        st.session_state.load2 = int(st.session_state.load2 * 0.6)
    elif routing == "TIER_2_L1":          
        st.session_state.load1 = int(st.session_state.load1 * 0.6)
    elif routing == "TIER_3":
        st.session_state.load1 = int(st.session_state.load1 * 0.7)
        st.session_state.load2 = int(st.session_state.load2 * 0.7)

    st.session_state.temperature = max(20, st.session_state.temperature - (drop * 0.5))
    st.session_state.applied_routing = routing
    st.session_state.apply_action = False
    st.session_state.show_success = True
    
    # Lock the button to prevent spamming!
    st.session_state.action_applied_once = True

# --- UNLOCK BUTTON WHEN SLIDERS ARE MOVED ---
def reset_action_state():
    st.session_state.action_applied_once = False
    st.session_state.apply_action = False
    st.session_state.show_success = False

def apply_scenario():
    """Updates sliders and loads automatically when a scenario is selected"""
    mode = st.session_state.demo_mode
    st.session_state.applied_routing = "NORMAL"
    st.session_state.action_applied_once = False
    st.session_state.current_routing = "NORMAL"
    st.session_state.expected_drop = 0
    st.session_state.apply_action = False

    if mode == "Normal":
        st.session_state.temperature = 72
        st.session_state.load1 = 65
        st.session_state.load2 = 40
    elif mode == "High Load":
        st.session_state.temperature = 88
        st.session_state.load1 = 90
        st.session_state.load2 = 60
    elif mode == "Critical":
        st.session_state.temperature = 105
        st.session_state.load1 = 110
        st.session_state.load2 = 80

# ------------------ LOAD DUMMY ML MODEL ------------------
class PrototypeMLModel:
    def predict(self, X):
        # We now use standard Pandas methods (.iloc) to read the DataFrame!
        load = X['Total_Load_kW'].iloc[0]
        ambient = X['Ambient_Temp_C'].iloc[0]

        # A simple fake physics formula to make the graph look realistic
        predicted_heat = 50 + (load * 0.12) + ((ambient - 30) * 1.5)
        return [predicted_heat]

ml_model = PrototypeMLModel()

st.title("NITW Smart Grid Digital Twin")

# ------------------ SIDEBAR ------------------
st.sidebar.title("Control Panel")
st.sidebar.selectbox("Scenario", ["Normal", "High Load", "Critical"], key="demo_mode", on_change=apply_scenario)
st.sidebar.markdown("---")

# Added on_change callbacks to unlock the button if a judge manually tests the sliders
ambient_temp = st.sidebar.slider("Ambient Temp (°C)", 0, 100, 39, key="amb_temp", on_change=reset_action_state)
st.sidebar.slider("Transformer A Temp (°C)", 20, 120, key="temperature", on_change=reset_action_state)

st.sidebar.slider("Load 1 (kW)", 0, 150, key="load1", on_change=reset_action_state)
st.sidebar.slider("Load 2 (kW)", 0, 150, key="load2", on_change=reset_action_state)

battery_soc = st.sidebar.slider("Battery SOC (%)", 0, 100, 70, key="bat_soc", on_change=reset_action_state)
lib_temp = st.sidebar.slider("Transformer B Temp (°C)", 30, 100, 65, key="lib_temp", on_change=reset_action_state)
lib_load = st.sidebar.number_input("Transformer B Load (kW)", 0, 200, 50, key="lib_load", on_change=reset_action_state)

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["Real-time grid", "Capacity simulator"])

with tab1:
    render_realtime_grid(ambient_temp, battery_soc, lib_temp, lib_load)

with tab2:
    # Passing the dummy model to your friend's feature 2 function
    render_feature2(ml_model)