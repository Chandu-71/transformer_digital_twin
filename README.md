# AI-Based Smart Grid Digital Twin

An intelligent transformer monitoring and decision-support system that combines **real-time prediction** and **capacity planning** using AI/ML.

## Live Demo

Visit: [https://mwmasters.streamlit.app](https://mwmasters.streamlit.app)

---

## Overview

This project addresses critical challenges in power systems:
- Transformer overloading
- Insulation degradation
- Inefficient infrastructure planning

We developed a **Digital Twin Dashboard** that enables:
- Predictive load balancing
- Real-time decision-making
- AI-driven capacity planning

---

## Key Features

### Feature 1: Predictive Grid Balancer
- Predicts future transformer temperature using ML
- Detects overload risk (Tier-based system)
- Dynamically reroutes loads:
  - Transformer B
  - Solar / BESS
- Estimates **insulation aging (cellulose degradation)**
- Provides real-time AI recommendations

---

### Feature 2: AI Capacity Planner
- Simulates 24-hour load profiles
- Predicts transformer stress under new loads
- Recommends optimal solution:
  - Load shifting
  - Solar + Battery (BESS)
  - Transformer upgrade
- Includes cost comparison (AI vs traditional)

---

## Role of AI/ML

AI enables the system to move beyond reactive monitoring by:
- Predicting future transformer conditions
- Detecting overload before it happens
- Optimizing load distribution in real-time
- Simulating future scenarios for planning
- Recommending cost-effective and safe solutions

---

## Tech Stack

- **Frontend/UI:** Streamlit, HTML, CSS
- **Visualization:** Plotly
- **Backend:** Python
- **Machine Learning:** Scikit-learn (Random Forest)
- **Data Handling:** Pandas, NumPy

---

## How to Run

```bash
pip install streamlit pandas numpy plotly scikit-learn
streamlit run main_app.py
```

---

## Screenshots

> <img width="1919" height="1079" alt="Screenshot 2026-03-22 134536" src="https://github.com/user-attachments/assets/2e18c262-16c5-40a1-a18a-db018d7b8a74" />
> <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ab6c10a7-fc8b-45f2-806b-07f0de3ae65c" />
> <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f8805f38-1a25-4b61-8f1f-dab738be1b14" />
> <img width="1919" height="843" alt="image" src="https://github.com/user-attachments/assets/583f0883-49b6-4584-9e3a-c20a50378d8d" />
