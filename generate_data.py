import pandas as pd
import numpy as np

print("⚙️ Generating New Data with Ambient Temperature...")
np.random.seed(42)
num_samples = 10000

# 1. AMBIENT WEATHER (20°C to 45°C)
ambient_temp = np.random.uniform(20, 45, num_samples)

# 2. ELECTRICAL LOAD (10kW to 120kW)
current_load_kW = np.random.uniform(10, 120, num_samples)

# 3. CURRENT TRANSFORMER TEMPERATURE
# Heating formula: Outside weather + Heat from electrical load
current_temp = ambient_temp + (current_load_kW * 0.4) + np.random.normal(0, 2, num_samples)

# 4. FUTURE TEMPERATURE (The Target)
future_temp = current_temp + (current_load_kW * 0.25) - ((current_temp - ambient_temp) * 0.1) + np.random.normal(0, 1.5, num_samples)

# Add Critical Meltdowns for AI Training
meltdown_indices = np.random.choice(num_samples, size=300, replace=False)
current_load_kW[meltdown_indices] = np.random.uniform(100, 130, 300)
current_temp[meltdown_indices] = np.random.uniform(85, 95, 300)
future_temp[meltdown_indices] = current_temp[meltdown_indices] + np.random.uniform(15, 25, 300)

# Combine into DataFrame
df = pd.DataFrame({
    'Ambient_Temp_C': np.round(ambient_temp, 1),
    'Total_Load_kW': np.round(current_load_kW, 1),
    'Current_Temp_C': np.round(current_temp, 1),
    'Future_Temp_30Min_C': np.round(future_temp, 1)
})

df.to_csv('transformer_training_data.csv', index=False)
print("✅ Success! New 'transformer_training_data.csv' created with Ambient Weather.")