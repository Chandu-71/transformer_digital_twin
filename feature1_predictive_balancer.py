import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ------------------ ML MODEL ENGINE ------------------
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('transformer_training_data.csv')
        x_features = df[['Current_Temp_C', 'Total_Load_kW', 'Ambient_Temp_C']]
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(x_features, df['Future_Temp_30Min_C'])
        return model
    except FileNotFoundError:
        class DummyModel:
            def predict(self, df):
                return (df['Current_Temp_C'] + (df['Total_Load_kW'] * 0.08)).values
        return DummyModel()

def run_ml_model(core_temp, total_load_a, ambient, tx_b_temp, tx_b_load, l1, l2):
    model = train_model()

    def predict_smooth(c_temp, l_kw, a_temp):
        df = pd.DataFrame([[c_temp, l_kw, a_temp]], columns=['Current_Temp_C', 'Total_Load_kW', 'Ambient_Temp_C'])
        base_pred = model.predict(df)[0]
        return base_pred + (l_kw * 0.015)

    pred_temp = predict_smooth(core_temp, total_load_a, ambient)

    routing_status, action = "NORMAL", "System stable. No action required."
    title, subtitle = "System Stable", "All loads operating within safe limits"
    bg, border, text_col = "#d1e7dd", "#badbcc", "#0f5132"
    expected_drop = 0

    if pred_temp >= 80:
        b_temp = tx_b_temp * 0.9 
        
        future_b_with_l2 = predict_smooth(b_temp, tx_b_load + l2, ambient)
        future_b_with_l1 = predict_smooth(b_temp, tx_b_load + l1, ambient)

        drop_with_l2 = max(0, pred_temp - predict_smooth(core_temp, total_load_a - l2, ambient))
        drop_with_l1 = max(0, pred_temp - predict_smooth(core_temp, total_load_a - l1, ambient))

        thermal_l2_safe = future_b_with_l2 < 95
        thermal_l1_safe = future_b_with_l1 < 95
        
        TX_B_MAX_CAPACITY = 130 
        electrical_l2_safe = (tx_b_load + l2) <= TX_B_MAX_CAPACITY
        electrical_l1_safe = (tx_b_load + l1) <= TX_B_MAX_CAPACITY
        
        l2_safe = thermal_l2_safe and electrical_l2_safe
        l1_safe = thermal_l1_safe and electrical_l1_safe

        if core_temp >= 105 or pred_temp >= 110:
            routing_status = "TIER_3"
            title, subtitle = "Critical Condition", "Immediate load shedding required"
            bg, border, text_col = "#f8d7da", "#f5c2c7", "#842029"
            action = "Emergency shedding. Disconnect non-critical loads immediately."
            expected_drop = 15.0
        elif l1_safe and l2_safe:
            if drop_with_l1 > drop_with_l2:
                routing_status = "TIER_1_L1"
                title, subtitle = "Tier 1: Optimal Balancing", f"Shifting Load 1 ({l1}kW) to Transformer B"
                bg, border, text_col = "#cff4fc", "#b6effb", "#055160"
                action = "Shift Load 1 to Transformer B for maximum cooling effect."
                expected_drop = round(drop_with_l1, 1)
            else:
                routing_status = "TIER_1_L2"
                title, subtitle = "Tier 1: Optimal Balancing", f"Shifting Load 2 ({l2}kW) to Transformer B"
                bg, border, text_col = "#fff3cd", "#ffecb5", "#664d03"
                action = "Shift Load 2 to Transformer B for optimal grid balance."
                expected_drop = round(drop_with_l2, 1)
        elif l2_safe:
            routing_status = "TIER_1_L2"
            title, subtitle = "Tier 1: Load Balancing", f"Shifting Load 2 ({l2}kW) to Transformer B"
            bg, border, text_col = "#fff3cd", "#ffecb5", "#664d03"
            action = "Shift Load 2 to Transformer B."
            expected_drop = round(drop_with_l2, 1)
        elif l1_safe:
            routing_status = "TIER_1_L1"
            title, subtitle = "Tier 1: Load Balancing", f"Shifting Load 1 ({l1}kW) to Transformer B"
            bg, border, text_col = "#fff3cd", "#ffecb5", "#664d03"
            action = "Shift Critical Load 1 to Transformer B."
            expected_drop = round(drop_with_l1, 1)
        
        # --- NEW: DYNAMIC SOLAR FALLBACK ---
        else:
            if drop_with_l1 > drop_with_l2:
                routing_status = "TIER_2_L1"
                title, subtitle = "Tier 2: Battery Support Active", "Node B full. Shifting to Solar/BESS."
                bg, border, text_col = "#fff3cd", "#ffecb5", "#664d03"
                action = f"Deploy Solar Array to absorb Load 1 ({l1}kW)."
                expected_drop = round(drop_with_l1, 1)
            else:
                routing_status = "TIER_2_L2"
                title, subtitle = "Tier 2: Battery Support Active", "Node B full. Shifting to Solar/BESS."
                bg, border, text_col = "#fff3cd", "#ffecb5", "#664d03"
                action = f"Deploy Solar Array to absorb Load 2 ({l2}kW)."
                expected_drop = round(drop_with_l2, 1)

    return {
        "pred_temp": pred_temp, "routing_status": routing_status, "title": title, "subtitle": subtitle,
        "bg_color": bg, "border_color": border, "text_color": text_col, "recommended_action": action,
        "anomaly_score": min(1.0, (pred_temp / 120)), "expected_temp_drop": expected_drop
    }

def get_connection(routing, load_type):
    if routing == "NORMAL":
        return "A"
    elif routing == "TIER_1_L2":
        return "B" if load_type == "L2" else "A"
    elif routing == "TIER_1_L1":
        return "B" if load_type == "L1" else "A"
    
    # --- NEW: DYNAMIC SOLAR CONNECTIONS ---
    elif routing == "TIER_2_L2":
        return "A_AND_SOLAR" if load_type == "L2" else "A"
    elif routing == "TIER_2_L1":
        return "A_AND_SOLAR" if load_type == "L1" else "A"
        
    elif routing == "TIER_3":
        return "DISCONNECTED" 
    else:
        return "A"

# ------------------ UI RENDER FUNCTION ------------------
def render_realtime_grid(ambient_temp, battery_soc, lib_temp, lib_load):
    st.subheader("Predictive Grid Balancer")

    current_temp = st.session_state.temperature
    current_load1 = st.session_state.load1
    current_load2 = st.session_state.load2
    total_load = current_load1 + current_load2

    ml_outputs = run_ml_model(current_temp, total_load, ambient_temp, lib_temp, lib_load, current_load1, current_load2)
    pred_temp = ml_outputs["pred_temp"]

    aging_factor = 2 ** ((pred_temp - 98) / 6)
    if aging_factor < 1:
        health_status = "Normal Aging"
        health_color = "#00c853"
    elif aging_factor < 3:
        health_status = "Accelerated Aging"
        health_color = "#ff9800"
    else:
        health_status = "Severe Degradation"
        health_color = "#ff1744"
        
    routing = ml_outputs["routing_status"]
    st.session_state.current_routing = routing
    st.session_state.expected_drop = ml_outputs["expected_temp_drop"]

    display_routing = st.session_state.get("applied_routing", "NORMAL")
    l1_conn = get_connection(display_routing, "L1")
    l2_conn = get_connection(display_routing, "L2")

    alert_section, main_section = st.columns([1, 0.01])[0], st.container()

    with alert_section:
        html_alert = f"""<div style='background-color:{ml_outputs["bg_color"]}; border:1px solid {ml_outputs["border_color"]}; padding:18px; border-radius:10px; display:flex; justify-content:space-between; align-items:center; color:{ml_outputs["text_color"]}; margin-bottom:15px;'>
            <div>
                <div style='font-size:18px; font-weight:600;'>{ml_outputs["title"]}</div>
                <div style='font-size:14px; opacity:0.8;'>{ml_outputs["subtitle"]}</div>
            </div>
        </div>"""
        st.markdown(html_alert, unsafe_allow_html=True)

    if routing != display_routing:
        if not st.session_state.get("action_applied_once"):
            st.warning("⚠️ AI Recommendation not yet applied to the grid. Click 'Apply AI Action' to rewire.")

    with main_section:
        left_col, right_col = st.columns([7, 3])

        with left_col:
            st.subheader("Digital Twin for Transformer A")
            flow_color = "#3b82f6" if total_load < 100 else "#ef4444"

            if current_temp < 80:
                t_bg, t_border, t_text, t_text_color = "#d1e7dd", "#20c997", "Safe zone", "#0f5132"
            elif current_temp < 95:
                t_bg, t_border, t_text, t_text_color = "#fff3cd", "#ffcc00", "Warning zone", "#664d03"
            else:
                t_bg, t_border, t_text, t_text_color = "#f8d7da", "#dc3545", "Danger zone", "#842029"

            def v_line_to_load(conn, target, color):
                is_active = target in conn
                line_style = f"2px solid {color}" if is_active else "2px dashed #adb5bd"
                arrow_color = color if is_active else "transparent"
                return f'<div style="width: 0px; height: 30px; border-left: {line_style}; position: relative;"><div style="position: absolute; bottom: -4px; left: -5px; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid {arrow_color};"></div></div>'

            def h_line_to_load(conn, target, color, side="right"):
                is_active = target in conn
                line_style = f"2px solid {color}" if is_active else "2px dashed #adb5bd"
                arrow_color = color if is_active else "transparent"
                if side == "right":
                    return f'<div style="position: absolute; left: 100%; top: 50%; width: 70px; border-top: {line_style}; z-index: 1;"><div style="position: absolute; right: -2px; top: -5px; width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 6px solid {arrow_color};"></div></div>'
                else:
                    return f'<div style="position: absolute; right: 100%; top: 50%; width: 70px; border-top: {line_style}; z-index: 1;"><div style="position: absolute; left: -2px; top: -5px; width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-right: 6px solid {arrow_color};"></div></div>'

            sld_html = f"""<div style="background: #ffffff; padding: 30px; border-radius: 12px; border: 1px solid #e5e7eb; font-family: sans-serif; color: #1f2937; display: flex; flex-direction: column; align-items: center; width: 100%; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); overflow-x: auto;">
                <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 12px 24px; text-align: center; background: #f8f9fa;">
                    <div style="font-weight: bold; font-size: 14px;">Main Power Grid</div>
                    <div style="font-size: 12px; color: #6b7280; margin-top: 2px;">33 kV incoming</div>
                </div>
                <div style="width: 2px; height: 30px; background: {flow_color}; position: relative;">
                    <div style="position: absolute; bottom: -2px; left: -4px; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid {flow_color};"></div>
                </div>
                <div style="background: {t_bg}; border: 6px solid {t_border}; border-radius: 8px; padding: 14px 40px; text-align: center; color: {t_text_color}; z-index: 2;">
                    <div style="font-weight: 600; font-size: 15px;">Transformer A</div>
                    <div style="font-size: 12px; margin-top: 2px; opacity: 0.9;">{current_temp}°C · {t_text}</div>
                </div>
                <div style="width: 2px; height: 25px; background: #adb5bd;"></div>
                <div style="width: 380px; height: 1px; background: #adb5bd; position: relative;">
                    <div style="position: absolute; left: 10px; top: 8px; font-size: 12px; color: #6b7280;">F1</div>
                    <div style="position: absolute; right: 10px; top: 8px; font-size: 12px; color: #6b7280;">F2</div>
                </div>
                <div style="display: flex; justify-content: space-between; width: 500px; align-items: flex-start; margin-top: 0px;">
                    <div style="display: flex; flex-direction: column; align-items: center; width: 120px; z-index: 2;">
                        {v_line_to_load(l1_conn, "A", flow_color)}
                        <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 14px 10px; text-align: center; background: #f8f9fa; width: 100%; box-sizing: border-box; height: 75px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-weight: bold; font-size: 14px;">Load 1</div>
                            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">{current_load1} kW</div>
                        </div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; width: 120px; padding-top: 30px; gap: 15px; z-index: 1;">
                        <div style="position: relative; width: 100%; border: 2px solid #0d6efd; border-radius: 6px; padding: 8px 0; text-align: center; background: #e6f0ff; font-size: 11px; font-weight: bold; color: #0d6efd;">
                            Transformer B
                            {h_line_to_load(l1_conn, "B", "#0d6efd", "left")}
                            {h_line_to_load(l2_conn, "B", "#0d6efd", "right")}
                        </div>
                        <div style="position: relative; width: 100%; border: 2px solid #f59e0b; border-radius: 6px; padding: 8px 0; text-align: center; background: #fef3c7; font-size: 11px; font-weight: bold; color: #b45309;">
                            Solar Array
                            {h_line_to_load(l1_conn, "SOLAR", "#f59e0b", "left")}
                            {h_line_to_load(l2_conn, "SOLAR", "#f59e0b", "right")}
                        </div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; width: 120px; z-index: 2;">
                        {v_line_to_load(l2_conn, "A", flow_color)}
                        <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 14px 10px; text-align: center; background: #f8f9fa; width: 100%; box-sizing: border-box; height: 75px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-weight: bold; font-size: 14px;">Load 2</div>
                            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">{current_load2} kW</div>
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 20px; margin-top: 40px; font-size: 12px; color: #6b7280; align-items: center; flex-wrap: wrap; justify-content: center;">
                    <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 20px; height: 2px; background: #3b82f6;"></div> Normal flow</div>
                    <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 20px; height: 2px; background: #ef4444;"></div> Heavy load</div>
                    <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 20px; height: 2px; background: #0d6efd;"></div> Trans B Active</div>
                    <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 20px; height: 2px; background: #f59e0b;"></div> Solar Active</div>
                    <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 20px; height: 2px; border-bottom: 2px dashed #adb5bd;"></div> Disconnected</div>
                </div>
            </div>"""
            st.markdown(sld_html, unsafe_allow_html=True)

            bg = ml_outputs["bg_color"]
            border = ml_outputs["border_color"]
            text_col = ml_outputs["text_color"]

            recommendation_html = f"""<div style="background:{bg}; border:1px solid {border}; padding:30px 24px; border-radius:12px; margin-top:20px; margin-bottom:25px; color:{text_col}; font-family:sans-serif; box-shadow:0 4px 12px rgba(0,0,0,0.05); width:100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                    <div style="font-size:22px; font-weight:600; display:flex; align-items:center; gap:8px;">AI Decision Engine</div>
                    <div style="font-size:14px; background:rgba(0,0,0,0.05); padding:6px 12px; border-radius:20px; border:1px solid rgba(0,0,0,0.1);">
                        Anomaly Score: <b>{ml_outputs["anomaly_score"]:.2f}</b>
                    </div>
                </div>
                <div style="font-size:16px; opacity:0.9; margin-bottom:24px;">
                    System Risk Level: <span style="font-weight:bold; font-size:18px;">{routing.replace('TIER_', 'LEVEL ')}</span>
                </div>
                <div style="height:1px; background:{border}; opacity:0.8; margin:24px 0;"></div>
                <div style="margin-bottom:24px;">
                    <div style="font-size:14px; opacity:0.8; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; font-weight:600;">Recommended Action</div>
                    <div style="font-size:20px; font-weight:500; line-height:1.5;">{ml_outputs["recommended_action"]}</div>
                </div>
                <div style="font-size:16px; background:rgba(255,255,255,0.4); padding:16px; border-radius:8px; border-left:4px solid {text_col}; display:flex; align-items:center; gap:10px;">
                    <b>Expected Impact:</b> Temperature drop of {ml_outputs["expected_temp_drop"]}°C within 30 mins
                </div>
            </div>"""
            st.markdown(recommendation_html, unsafe_allow_html=True)

            if st.session_state.get("show_success"):
                st.success("⚡ Action applied: Load redistributed and temperature dropping.")
                st.session_state.show_success = False

            if routing != "NORMAL":
                if not st.session_state.get("action_applied_once"):
                    if st.button("⚡ Apply AI Action", type="primary", use_container_width=True):
                        st.session_state.apply_action = True
                        st.rerun()
                else:
                    st.button("⏳ Action Applied (Grid Stabilizing...)", type="primary", disabled=True, use_container_width=True)
                    st.info("💡 **Demo Tip:** The AI is locked to prevent button spamming. To unlock it, manually lower the **Transformer A Temp** slider to simulate the grid cooling down over time!")
            else:
                st.button("✅ System Optimized", disabled=True, use_container_width=True)

        with right_col:
            st.subheader("System Metrics")
            fig_temp = go.Figure(go.Indicator(
                mode="gauge+number", value=ml_outputs["pred_temp"],
                title={"text": "Predicted Temp (30 min)"},
                gauge={"axis": {"range": [0, 120]}, "bar": {"color": "#00eaa4"},
                       "steps": [{"range": [0, 80], "color": "#009e54"},
                                 {"range": [80, 95], "color": "#d49f02"},
                                 {"range": [95, 120], "color": "#be0013"}]}
            ))
            fig_temp.update_layout(height=300, margin=dict(t=30, b=0))
            with st.container(border=True):
                st.plotly_chart(fig_temp, use_container_width=True)

            with st.container(border=True):
                st.markdown("### Insulation Health")
                html_card = f"""<div style="background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 2px 5px rgba(0,0,0,0.03); margin: 12px 0; text-align: center; font-family: sans-serif;">
                    <div style="display: flex; justify-content: center; align-items: center; gap: 8px; font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                        Cellulose Aging Factor
                    </div>
                    <div style="font-size: 42px; font-weight: 800; color: {health_color}; margin: 16px 0 12px 0; line-height: 1;">
                        {aging_factor:.2f}×
                    </div>
                    <div style="display: inline-block; border: 1px solid {health_color}; color: {health_color}; background: rgba(0,0,0,0.02); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;">
                        {health_status}
                    </div>
                </div>"""
                st.markdown(html_card, unsafe_allow_html=True)

            fig_load = go.Figure(go.Indicator(
                mode="gauge+number", value=total_load, title={"text": "Current Load (kW)"},
                gauge={"axis": {"range": [0, 200]}, "bar": {"color": "#4dabf7"}}
            ))
            fig_load.update_layout(height=300, margin=dict(t=30, b=0))
            with st.container(border=True):
                st.plotly_chart(fig_load, use_container_width=True)

            with st.container(border=True):
                st.markdown("### Battery SOC")
                st.progress(battery_soc / 100)
                st.write(f"{battery_soc}%")