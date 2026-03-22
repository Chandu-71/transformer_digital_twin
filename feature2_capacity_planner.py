import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_feature2(ml_model):
    st.markdown("### AI Capacity Planner")

    sim_left, sim_right = st.columns([35, 65])

    # ---------------- LEFT PANEL: INPUTS ----------------
    with sim_left:
        st.markdown("""<div style="background: #1e293b; padding:15px; border-radius:10px 10px 0 0; color:white;">
                <h4 style="margin:0; font-size:16px;">Step 1: Define New Load Profile</h4>
            </div>""", unsafe_allow_html=True)

        with st.container(border=True):

            # --- Choose between Built-in or CSV Upload ---
            load_source = st.radio("Select Load Pattern Source:",
                                   ["Built-in Profiles", "Upload Custom CSV"],
                                   horizontal=True)

            new_load_curve = np.zeros(24)  # Default empty array

            if load_source == "Built-in Profiles":
                profile_type = st.selectbox(
                    "Choose a load scenario:",
                    # FIX: Removed the leading space before "DANGER ZONE"
                    ["Standard Daytime (e.g., IT Lab)", "Night Operations (e.g., Hostels)",
                     "DANGER ZONE (Heavy Industrial)"] 
                )

                # Setup the built-in arrays
                if profile_type == "Standard Daytime (e.g., IT Lab)":
                    new_load_curve[9:18] = 120
                elif profile_type == "Night Operations (e.g., Hostels)":
                    new_load_curve[18:24] = 90
                    new_load_curve[0:6] = 60
                elif profile_type == "DANGER ZONE (Heavy Industrial)":
                    new_load_curve[8:20] = 300  # Massive load to force hardware upgrade
                    st.error("Danger Zone active: Simulating severe localized grid stress.")

            else:
                # --- CSV Upload Feature ---
                st.info("CSV should contain 24 rows of numeric values (representing kW per hour).")
                uploaded_file = st.file_uploader("Upload 24-Hour Load CSV", type=['csv'])

                if uploaded_file is not None:
                    try:
                        df_custom = pd.read_csv(uploaded_file)
                        # Take the first column of the CSV and use its first 24 rows
                        new_load_curve = df_custom.iloc[:, 0].values[:24]
                        st.success("Custom load profile applied successfully!")
                    except Exception:
                        st.error("Error reading CSV. Ensure it has at least 24 numbers.")
                else:
                    st.warning("Waiting for CSV upload. Using 0 kW load.")

            st.markdown("---")

            # --- Demo Mode Expander ---
            with st.expander(" Demo Mode (Manual Grid Stress)", expanded=False):
                st.markdown("**Live Grid Override Controls**")
                stress_multiplier = st.slider("Global Base Load Scaling (%)", 50, 200, 100)
                peak_warp = st.slider("Manual Peak Intensity Multiplier", 0.5, 2.0, 1.0)

            run_sim = st.button(" Run AI Impact Analysis", type="primary", use_container_width=True)

    # ---------------- DATA & ML LOGIC CALCULATIONS ----------------
    hours = np.arange(0, 24)
    ambient_temp_24h = 30 + 12 * np.sin(np.pi * (hours - 8) / 12)

    # Base Load
    tx_a_base = (220 + 80 * np.sin(np.pi * (hours - 6) / 12) +
                 40 * np.sin(np.pi * (hours - 16) / 6)) * (stress_multiplier / 100)

    # Apply the peak warp multiplier from the hidden Demo Mode
    new_load_curve = new_load_curve * peak_warp

    tx_a_total_load = tx_a_base + new_load_curve

    predicted_temps_a = []
    for h in range(24):
        df = pd.DataFrame([[65.0, tx_a_total_load[h], ambient_temp_24h[h]]],
                          columns=['Current_Temp_C', 'Total_Load_kW', 'Ambient_Temp_C'])
        predicted_temps_a.append(ml_model.predict(df)[0])

    # ---------------- LEFT PANEL: AI DECISION ENGINE ----------------
    with sim_left:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("###  AI Capacity Recommendation")

        if not run_sim:
            st.info(" Click 'Run AI Impact Analysis' to generate AI recommendation")
        else:
            peak_demand = np.max(tx_a_total_load)
            transformer_limit = 400  # Max safe capacity

            if peak_demand <= transformer_limit:
                action_text = "Current setup is fine."
                impact_text = "Peak load is within safe operating capacity. No action needed."
            elif peak_demand <= transformer_limit + 30:
                overload = peak_demand - transformer_limit
                action_text = f"Shift ~{overload:.1f} kW of load to Transformer B."
                impact_text = "Prevents Transformer A overheating without buying new equipment."
            elif peak_demand <= transformer_limit + 100:
                required_solar = (peak_demand - transformer_limit) + 20
                action_text = f"Install a {required_solar:.0f} kW Rooftop Solar + BESS array."
                impact_text = "Offsets peak daylight loads and keeps transformer within safe limits."
            else:
                action_text = "Replace Transformer A with a higher capacity unit."
                impact_text = "Current overload is too high for solar or load shifting alone."

            decision_html = f"""<div style="background:#fdf6e3; padding:24px; border-radius:12px; border:1px solid #f1e2b3; color:#5c4c16; font-family:sans-serif; margin-bottom: 20px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
            <div style="font-size:12px; font-weight:700; color:#8a6d1c; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">
            Recommended Action
            </div>
            <div style="font-size:18px; font-weight:500; margin-bottom:16px;">
            {action_text}
            </div>
            <div style="background:#f9f1d8; padding:12px 16px; border-left:4px solid #d4b455; border-radius:4px; font-size:14px; display:flex; gap:8px;">
            <span style="font-weight:600;">Expected Impact:</span> {impact_text}
            </div>
            </div>"""
            st.markdown(decision_html, unsafe_allow_html=True)

    # ---------------- RIGHT PANEL: GRAPHS & COST COMPARISON ----------------
    with sim_right:
        st.markdown("####  Transformer A: Base Load vs New Total Demand")

        fig_a = go.Figure()

        # 1. Base Active Load (Grey)
        fig_a.add_trace(go.Scatter(
            x=hours, y=tx_a_base,
            name='Base Load (kW)',
            fill='tozeroy',
            line=dict(color='#9ca3af', width=2)
        ))

        # 2. Total Load (Base + New Load) (Red)
        fig_a.add_trace(go.Scatter(
            x=hours, y=tx_a_total_load,
            name='Total Load w/ Project (kW)',
            fill='tonexty',
            line=dict(color='#ef4444', width=3)
        ))

        fig_a.update_layout(
            height=300, margin=dict(t=20, b=20),
            yaxis=dict(title="Load (kW)"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_a, use_container_width=True)

        st.markdown("####  Transformer A: Capacity Utilization")
        utilization = (tx_a_total_load / 400) * 100

        fig_util = go.Figure()
        fig_util.add_trace(go.Scatter(x=hours, y=utilization, fill='tozeroy', line=dict(color='#22c55e', width=3)))
        fig_util.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Max Capacity")
        fig_util.update_layout(height=200, margin=dict(t=20, b=20), yaxis=dict(title="Utilization (%)"))
        st.plotly_chart(fig_util, use_container_width=True)

        # -------- DYNAMIC COST CALCULATIONS & UI --------
        st.markdown("###  AI Financial & Cost Optimization")

        if not run_sim:
            st.info(" Click 'Run AI Impact Analysis' to calculate cost scenarios and view savings.")
        else:
            peak_demand = np.max(tx_a_total_load)
            transformer_limit = 400

            #  DYNAMIC LOGIC: Match Both Cost Boxes to the AI Recommendation
            if peak_demand <= transformer_limit:
                # 1. No action needed (Zero Cost)
                trad_cost = 0
                trad_desc = "Current setup handles the load safely. No traditional hardware upgrades required."
                trad_badges = '<span style="background:#6c757d; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Status Quo</span>'

                ai_cost = 0
                ai_desc = "Capacity is sufficient. AI continuous monitoring is active to prevent future unseen overloads."
                ai_badges = '<span style="background:#198754; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">System Optimized</span>'

            elif peak_demand <= transformer_limit + 30:
                # 2. Shift Load
                trad_cost = 500000
                trad_desc = "Standard procedure: Replace transformer early to handle minor peak overloads."
                trad_badges = '<span style="background:#dc3545; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">High cost</span> <span style="background:#6c757d; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Hardware</span>'

                ai_cost = 45000
                ai_desc = "Software-based load shifting. Routes excess demand without buying heavy hardware."
                ai_badges = f'<span style="background:#0dcaf0; color:black; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Smart Grid</span> <span style="background:#0f5132; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Saves ₹{(trad_cost - ai_cost):,}</span>'

            elif peak_demand <= transformer_limit + 100:
                # 3. Solar + Battery
                trad_cost = int(800000 * (peak_demand / 350))
                trad_desc = "Replace transformer with higher capacity unit. High upfront cost, no continuous optimization."
                trad_badges = '<span style="background:#dc3545; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">High cost</span> <span style="background:#6c757d; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Grid-dependent</span>'

                ai_cost = int(250000 + (peak_demand - 350) * 800)
                ai_desc = "Install rooftop solar + battery (BESS). Reduces peak load dynamically and improves efficiency."
                ai_badges = f'<span style="background:#198754; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Green energy</span> <span style="background:#0f5132; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Saves ₹{(trad_cost - ai_cost):,}</span>'

            else:
                # 4. Replace Transformer (Extreme Overload)
                trad_cost = int(800000 * (peak_demand / 350))
                trad_desc = "Replace transformer with higher capacity unit. Standard industry estimation."
                trad_badges = '<span style="background:#dc3545; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">High cost</span> <span style="background:#6c757d; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Grid-dependent</span>'

                ai_cost = int(trad_cost * 0.90)
                ai_desc = "Replace Transformer A. Overload is too high for solar. AI optimizes the exact capacity needed to save costs."
                ai_badges = f'<span style="background:#ffc107; color:black; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Hardware Upgrade</span> <span style="background:#0f5132; color:white; padding:5px 10px; border-radius:6px; font-size:12px; font-weight:600;">Saves ₹{(trad_cost - ai_cost):,}</span>'

            #  RENDER THE HTML
            cost_html = f"""<div style="display:flex; gap: 20px; margin-top: 10px; font-family:sans-serif;">
            <div style="flex:1; border:1px solid #f5c2c7; background:#fcf0f1; padding:24px; border-radius:12px; color:#842029; box-shadow:0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-weight:600; font-size:16px; display:flex; align-items:center; gap:6px;">Option A · Traditional </div>
            <div style="font-size:32px; font-weight:700; margin:12px 0;">₹{trad_cost:,}</div>
            <div style="font-size:14px; opacity:0.85; line-height:1.5;">{trad_desc}</div>
            <div style="margin-top:20px; display:flex; gap:10px;">
            {trad_badges}
            </div>
            </div>
            <div style="flex:1; border:1px solid #badbcc; background:#f0f9f4; padding:24px; border-radius:12px; color:#0f5132; box-shadow:0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-weight:600; font-size:16px; display:flex; align-items:center; gap:6px;">Option B · AI Recommended </div>
            <div style="font-size:32px; font-weight:700; margin:12px 0; color:#198754;">₹{ai_cost:,}</div>
            <div style="font-size:14px; opacity:0.85; line-height:1.5;">{ai_desc}</div>
            <div style="margin-top:20px; display:flex; gap:10px;">
            {ai_badges}
            </div>
            </div>
            </div>"""

            st.markdown(cost_html, unsafe_allow_html=True)