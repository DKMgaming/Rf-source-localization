import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from io import BytesIO
from math import atan2, degrees, radians, sin, cos, sqrt

# --- Hàm phụ ---
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    azimuth = (degrees(atan2(x, y)) + 360) % 360
    return azimuth

def simulate_signal_strength(dist_km, h, freq_mhz):
    path_loss = 32.45 + 20 * np.log10(dist_km + 0.1) + 20 * np.log10(freq_mhz + 1)
    return -30 - path_loss + 10 * np.log10(h + 1)

# --- Giao diện ---
st.title("Huấn luyện & Dự đoán tọa độ nguồn phát xạ")

tab1, tab2 = st.tabs(["1. Huấn luyện mô hình", "2. Dự đoán tọa độ"])

# --- Tab 1: Huấn luyện ---
with tab1:
    st.subheader("Sinh dữ liệu mô phỏng và huấn luyện mô hình")

    if st.button("Huấn luyện mô hình từ dữ liệu mô phỏng"):
        np.random.seed(42)
        n_samples = 500
        data = []
        for _ in range(n_samples):
            lat_tx = np.random.uniform(10.0, 21.0)
            lon_tx = np.random.uniform(105.0, 109.0)
            lat_rx = lat_tx + np.random.uniform(-0.1, 0.1)
            lon_rx = lon_tx + np.random.uniform(-0.1, 0.1)
            h_rx = np.random.uniform(5, 50)
            freq = np.random.uniform(400, 2600)

            azimuth = calculate_azimuth(lat_rx, lon_rx, lat_tx, lon_tx)
            distance = sqrt((lat_tx - lat_rx)**2 + (lon_tx - lon_rx)**2) * 111
            signal = simulate_signal_strength(distance, h_rx, freq)

            data.append({
                "lat_receiver": lat_rx,
                "lon_receiver": lon_rx,
                "antenna_height": h_rx,
                "azimuth": azimuth,
                "frequency": freq,
                "signal_strength": signal,
                "lat_emitter": lat_tx,
                "lon_emitter": lon_tx
            })

        df = pd.DataFrame(data)
        df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
        df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))

        X = df[['lat_receiver', 'lon_receiver', 'antenna_height', 'signal_strength', 'frequency', 'azimuth_sin', 'azimuth_cos']]
        y = df[['lat_emitter', 'lon_emitter']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.success(f"Huấn luyện xong - MAE: {mae:.6f}")

        # Lưu mô hình tạm
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        st.download_button(
            label="Tải mô hình huấn luyện (.joblib)",
            data=buffer,
            file_name="location_model_with_freq.joblib",
            mime="application/octet-stream"
        )

# --- Tab 2: Dự đoán ---
with tab2:
    st.subheader("Dự đoán tọa độ nguồn phát")

    uploaded_model = st.file_uploader("Tải lên mô hình đã huấn luyện (.joblib)", type=["joblib"])
    if uploaded_model:
        model = joblib.load(uploaded_model)

        lat_rx = st.number_input("Vĩ độ trạm thu", value=16.0)
        lon_rx = st.number_input("Kinh độ trạm thu", value=108.0)
        h_rx = st.number_input("Chiều cao anten (m)", value=30.0)
        signal = st.number_input("Mức tín hiệu thu (dBm)", value=-80.0)
        freq = st.number_input("Tần số (MHz)", value=900.0)
        azimuth = st.number_input("Góc phương vị (độ)", value=45.0)

        if st.button("Dự đoán tọa độ nguồn phát"):
            az_sin = np.sin(np.radians(azimuth))
            az_cos = np.cos(np.radians(azimuth))
            X_input = np.array([[lat_rx, lon_rx, h_rx, signal, freq, az_sin, az_cos]])
            prediction = model.predict(X_input)

            lat_pred, lon_pred = prediction[0]
            st.success("Tọa độ nguồn phát xạ dự đoán:")
            st.markdown(f"- **Vĩ độ**: `{lat_pred:.6f}`")
            st.markdown(f"- **Kinh độ**: `{lon_pred:.6f}`")

            st.map(pd.DataFrame({'lat': [lat_rx, lat_pred], 'lon': [lon_rx, lon_pred]}))
