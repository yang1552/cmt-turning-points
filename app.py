# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import pairwise_distances

# ---- Sidebar 설정 ----
maturity_options = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30"
}
selected_maturity = st.sidebar.selectbox("Select Treasury Maturity", list(maturity_options.keys()))
fred_id = maturity_options[selected_maturity]

# 분석 기준 선택
analysis_type = st.sidebar.radio("Select Analysis Type", ("Absolute Change (diff)", "Percent Change (pct_change)"))

# 데이터 로드
@st.cache_data
def load_data(fred_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date", fred_id: "Rate"})
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

df = load_data(fred_id)

# 날짜 설정
min_date, max_date = df["Date"].min(), df["Date"].max()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.stop()

# 파라미터
frac = st.sidebar.slider("LOESS Smoothing (frac)", 0.001, 0.2, 0.05, step=0.005)
threshold = st.sidebar.slider("Slope Threshold", 0.0001, 0.02, 0.005, step=0.0005, format="%.4f")
window = st.sidebar.slider("Turning Point Window (days)", 5, 90, 30, step=5)

# 필터링
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()

# LOESS 스무딩
smoothed = lowess(df["Rate"], df["Date"], frac=frac)
smoothed_dates = pd.to_datetime(smoothed[:, 0])
smoothed_values = smoothed[:, 1]
slopes = np.diff(smoothed_values)
slope_dates = smoothed_dates[1:]

# 전환점 후보 찾기
candidate_idxs = np.where((np.abs(slopes) > 0) & (np.abs(slopes) < threshold))[0]

def find_turning_points(values, candidate_idxs, window):
    peaks, troughs = [], []
    for idx in candidate_idxs:
        start = max(0, idx - window)
        end = min(len(values), idx + window + 1)
        segment = values[start:end]
        val = values[idx]
        if val == np.max(segment):
            peaks.append(idx)
        elif val == np.min(segment):
            troughs.append(idx)
    return peaks, troughs

peak_idxs, trough_idxs = find_turning_points(smoothed_values, candidate_idxs, window)

# 시각화
st.title(f"{selected_maturity} CMT Rate Turning Point Analyzer")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Rate'], label='Raw Rate', alpha=0.4)
ax.plot(smoothed_dates, smoothed_values, label='LOESS Smoothed', color='red')
ax.scatter(slope_dates[peak_idxs], smoothed_values[1:][peak_idxs], color='blue', label='Peaks')
ax.scatter(slope_dates[trough_idxs], smoothed_values[1:][trough_idxs], color='green', label='Troughs')
ax.legend(); ax.grid(True)
st.pyplot(fig)

# 분석 함수
def analyze_segment(df, dates, window, mode="diff"):
    rows = []
    for dt in dates:
        segment = df[(df["Date"] <= dt) & (df["Date"] > dt - pd.Timedelta(days=window))]
        if len(segment) < 2: continue
        if mode == "diff":
            values = segment["Rate"].diff().dropna()
            total = segment["Rate"].iloc[-1] - segment["Rate"].iloc[0]
        else:
            values = segment["Rate"].pct_change().dropna()
            total = (segment["Rate"].iloc[-1] - segment["Rate"].iloc[0]) / segment["Rate"].iloc[0]
        rows.append({
            "Turning Point": dt.date(),
            "Start Date": segment["Date"].min().date(),
            "End Date": segment["Date"].max().date(),
            "Mean Change": values.mean(),
            "Std Dev": values.std(),
            "Total Change": total,
            "Max Change": values.max(),
            "Min Change": values.min()
        })
    return pd.DataFrame(rows)

mode = "pct" if analysis_type == "Percent Change (pct_change)" else "diff"
peak_df = analyze_segment(df, slope_dates[peak_idxs], window, mode)
trough_df = analyze_segment(df, slope_dates[trough_idxs], window, mode)

# Control 분석
exclude_dates = set()
for idx in peak_idxs + trough_idxs:
    ref_date = slope_dates[idx]
    dates_to_exclude = pd.date_range(ref_date - pd.Timedelta(days=window), ref_date)
    exclude_dates.update(dates_to_exclude.date)

control_df = df[~df["Date"].dt.date.isin(exclude_dates)].copy()
rolling_stats = []
for i in range(len(control_df) - window):
    sub = control_df.iloc[i:i+window]
    if mode == "diff":
        values = sub["Rate"].diff().dropna()
        total = sub["Rate"].iloc[-1] - sub["Rate"].iloc[0]
    else:
        values = sub["Rate"].pct_change().dropna()
        total = (sub["Rate"].iloc[-1] - sub["Rate"].iloc[0]) / sub["Rate"].iloc[0]
    rolling_stats.append({
        "Window Start": sub["Date"].iloc[0].date(),
        "Window End": sub["Date"].iloc[-1].date(),
        "Mean Change": values.mean(),
        "Std Dev": values.std(),
        "Total Change": total,
        "Max Change": values.max(),
        "Min Change": values.min()
    })

control_df_stats = pd.DataFrame(rolling_stats)

# 테이블 표시
st.markdown("### 피크 분석")
st.dataframe(peak_df)
st.download_button("Download Peak Stats", data=peak_df.to_csv(index=False), file_name="peak_stats.csv")

st.markdown("### 트러프 분석")
st.dataframe(trough_df)
st.download_button("Download Trough Stats", data=trough_df.to_csv(index=False), file_name="trough_stats.csv")

st.markdown("### 커트롤 구간")
st.dataframe(control_df_stats)
st.download_button("Download Control Stats", data=control_df_stats.to_csv(index=False), file_name="control_stats.csv")

# 통계 비교
def compare_groups(label, a_df, b_df):
    if not a_df.empty and not b_df.empty:
        t_stat, p_val = ttest_ind(a_df["Total Change"], b_df["Total Change"], equal_var=False)
        st.write(f"#### {label} vs Control")
        st.write(f"Mean: {a_df['Total Change'].mean():.4f} vs {b_df['Total Change'].mean():.4f}")
        st.write(f"T-test p-value: {p_val:.4f}")
        if p_val < 0.05:
            st.success("📊 Statistically significant difference.")
        else:
            st.info("No statistically significant difference.")

compare_groups("Peak", peak_df, control_df_stats)
compare_groups("Trough", trough_df, control_df_stats)

# 분포 시각화
st.markdown("### 📊 Distribution of Changes")
fig2, ax2 = plt.subplots()
sns.kdeplot(peak_df["Total Change"], label="Peak", fill=True, ax=ax2)
sns.kdeplot(trough_df["Total Change"], label="Trough", fill=True, ax=ax2)
sns.kdeplot(control_df_stats["Total Change"], label="Control", fill=True, ax=ax2)
ax2.legend()
st.pyplot(fig2)

# 최신 구간 분석 및 비교
def get_mean_distance(target_df, ref_df):
    features = ["Mean Change", "Std Dev", "Total Change", "Max Change", "Min Change"]
    distances = pairwise_distances(target_df[features], ref_df[features], metric="euclidean")
    return np.mean(distances, axis=1)

latest = df.tail(window).copy()
if mode == "diff":
    changes = latest["Rate"].diff().dropna()
    total = latest["Rate"].iloc[-1] - latest["Rate"].iloc[0]
else:
    changes = latest["Rate"].pct_change().dropna()
    total = (latest["Rate"].iloc[-1] - latest["Rate"].iloc[0]) / latest["Rate"].iloc[0]

latest_summary = pd.DataFrame([{
    "Mean Change": changes.mean(),
    "Std Dev": changes.std(),
    "Total Change": total,
    "Max Change": changes.max(),
    "Min Change": changes.min()
}])

peak_dist = get_mean_distance(latest_summary, peak_df)
trough_dist = get_mean_distance(latest_summary, trough_df)
control_dist = get_mean_distance(latest_summary, control_df_stats)

best_match = np.argmin([peak_dist.mean(), trough_dist.mean(), control_dist.mean()])
best_label = ["Peak", "Trough", "Control"][best_match]
st.markdown("### 📌 Similarity of Most Recent Period")
st.write("The most recent period is most similar to:", f"**{best_label}**")
