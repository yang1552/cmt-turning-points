import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date", "DGS10": "Rate"})
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df.dropna(inplace=True)
    df = df[df["Date"] >= "2015-01-01"]
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

st.title("🧭 10Y CMT Rate Turning Point Analyzer")

# Sidebar for parameters
st.sidebar.header("🔧 Parameters")
frac = st.sidebar.slider("LOESS Smoothing (frac)", 0.01, 0.2, 0.05, step=0.005)
threshold = st.sidebar.slider("Slope Threshold", 0.0001, 0.02, 0.005, step=0.0005)
window = st.sidebar.slider("Window size (days)", 5, 90, 30, step=5)

# LOESS smoothing
smoothed = lowess(df['Rate'], df['Date'], frac=frac)
smoothed_dates = pd.to_datetime(smoothed[:, 0])
smoothed_values = smoothed[:, 1]

# Slope calculation
slopes = np.diff(smoothed_values)
slope_dates = smoothed_dates[1:]
candidate_idxs = np.where((np.abs(slopes) > 0) & (np.abs(slopes) < threshold))[0]

# Peak & Trough Detection
peak_idxs, trough_idxs = [], []
for idx in candidate_idxs:
    start = max(0, idx - window)
    end = min(len(smoothed_values), idx + window + 1)
    segment = smoothed_values[start:end]
    value = smoothed_values[idx]
    if value == np.max(segment):
        peak_idxs.append(idx)
    elif value == np.min(segment):
        trough_idxs.append(idx)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Rate'], label='Raw 10Y Rate', alpha=0.4)
ax.plot(smoothed_dates, smoothed_values, color='red', label='LOESS Smoothed')
ax.scatter(slope_dates[peak_idxs], smoothed_values[1:][peak_idxs], color='blue', label='Peaks')
ax.scatter(slope_dates[trough_idxs], smoothed_values[1:][trough_idxs], color='green', label='Troughs')
ax.set_title("10Y CMT Rate: Turning Points (LOESS)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("""
---
🔎 **Parameter Descriptions:**

- **LOESS frac**: Controls how smooth the red line is (higher = smoother)
- **Slope Threshold**: Lower = only flatter areas become turning points
- **Window size**: Number of days before/after to confirm local peak/trough
""")
