# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import datetime

# 만기 선택: FRED 코드 매핑
maturity_options = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30"
}

selected_maturity = st.sidebar.selectbox("Select Treasury Maturity", list(maturity_options.keys()))
fred_id = maturity_options[selected_maturity]
# 데이터 불러오기 함수
@st.cache_data
def load_data(fred_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date", fred_id: "Rate"})
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# 데이터 불러오기
df = load_data(fred_id)

# 날짜 선택 범위 설정
min_date = df["Date"].min()
max_date = df["Date"].max()
default_start = pd.to_datetime("2015-01-01")
default_end = max_date

# Sidebar - 사용자 설정 입력
st.sidebar.header("🔧 Parameters")



# 날짜 선택
start_date = st.sidebar.date_input(
    "Select chart start date",
    value=default_start,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "Select chart end date",
    value=default_end,
    min_value=min_date,
    max_value=max_date
)

# 시작일과 종료일의 유효성 체크
if start_date > end_date:
    st.error("🚫 End date must be after start date.")
    st.stop()

# 사용자 입력 파라미터
frac = st.sidebar.slider("LOESS Smoothing (frac)", 0.001, 0.2, 0.05, step=0.005)
threshold = st.sidebar.slider("Slope Threshold", 0.0001, 0.02, 0.005, step=0.0005)
window = st.sidebar.slider("Turning Point Window (days)", 5, 90, 30, step=5)

# 분석용 데이터 필터링
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()

# LOESS 스무딩
smoothed = lowess(df['Rate'], df['Date'], frac=frac)
smoothed_dates = pd.to_datetime(smoothed[:, 0])
smoothed_values = smoothed[:, 1]

# 기울기 계산
slopes = np.diff(smoothed_values)
slope_dates = smoothed_dates[1:]

# 전환점 후보: 기울기 절댓값이 threshold보다 작은 지점
candidate_idxs = np.where((np.abs(slopes) > 0) & (np.abs(slopes) < threshold))[0]

# 전환점 검출 (전/후 window 기간 내에서 최댓값 → Peak, 최솟값 → Trough)
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

# Streamlit 메인 타이틀
st.title(f"📈 {selected_maturity} CMT Rate Turning Point Analyzer")

# 차트 출력
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Rate'], label=f'Raw {selected_maturity} Rate', alpha=0.4)
ax.set_title(f"{selected_maturity} CMT Rate: Turning Points (LOESS)")
ax.plot(smoothed_dates, smoothed_values, color='red', label='LOESS Smoothed')
ax.scatter(slope_dates[peak_idxs], smoothed_values[1:][peak_idxs], color='blue', label='Peaks')
ax.scatter(slope_dates[trough_idxs], smoothed_values[1:][trough_idxs], color='green', label='Troughs')
ax.set_title("10Y CMT Rate: Turning Points (LOESS)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 설명 텍스트
st.markdown("""
---
### ℹ️ Parameters Description

- **Chart Start / End Date**: 분석할 날짜 범위를 선택합니다  
- **LOESS Smoothing (frac)**: 빨간 곡선을 얼마나 부드럽게 그릴지 조정합니다  
- **Slope Threshold**: 기울기 변화가 얼마나 작아야 전환점으로 간주할지 설정합니다  
- **Window Size**: 전환점이 진짜인지 확인하기 위해 앞뒤 며칠을 비교할지 설정합니다  
""")

def get_surrounding_data(idx_list, smoothed_dates, smoothed_values, window):
    rows = []
    n = len(smoothed_values)
    for idx in idx_list:
        center_date = smoothed_dates[idx].date()
        center_value = smoothed_values[idx]

        prev_idx = max(0, idx - window)
        next_idx = min(n - 1, idx + window)

        prev_date = smoothed_dates[prev_idx].date()
        prev_value = smoothed_values[prev_idx]

        next_date = smoothed_dates[next_idx].date()
        next_value = smoothed_values[next_idx]

        rows.append({
            "Turning Point Date": center_date,
            "Turning Point Rate": center_value,
            f"{window} Days Before Date": prev_date,
            f"{window} Days Before Rate": prev_value,
            f"{window} Days After Date": next_date,
            f"{window} Days After Rate": next_value
        })
    return rows

# Peak, Trough 데이터 생성 시 window 슬라이더 값 전달
peak_data = get_surrounding_data(peak_idxs, smoothed_dates, smoothed_values, window)
trough_data = get_surrounding_data(trough_idxs, smoothed_dates, smoothed_values, window)
# Peak 테이블 생성
peak_data = get_surrounding_data(peak_idxs, smoothed_dates, smoothed_values, window=30)
# Through 테이블 생성
trough_data = get_surrounding_data(trough_idxs, smoothed_dates, smoothed_values, window=30)

# Streamlit 출력
st.markdown("### 🔹 Peak Turning Points Details")
if peak_data:
    st.dataframe(pd.DataFrame(peak_data))
else:
    st.write("No Peak turning points detected with current parameters.")

st.markdown("### 🔹 Trough Turning Points Details")
if trough_data:
    st.dataframe(pd.DataFrame(trough_data))
else:
    st.write("No Trough turning points detected with current parameters.")
