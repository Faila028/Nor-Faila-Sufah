# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# ==========================================
# CONFIG HALAMAN
# ==========================================

st.set_page_config(
    page_title="Forecasting Barang",
    page_icon="📦",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main { background-color: #f5f7fa; }

h1 {
    color: #1f4e79;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #1f4e79;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

.stDownloadButton>button {
    background-color: #28a745;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("📦 Clustering & Forecasting Permintaan Barang")

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df.head())

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])
    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    pivot_table = df.pivot_table(
        index='id_produk',
        columns='Bulan',
        values='keluar',
        aggfunc='sum',
        fill_value=0
    )

    urutan_bulan = [
        'Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23',
        'Jul-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24','May-24','Jun-24',
        'Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(columns=urutan_bulan)

    st.dataframe(pivot_table)

    # ==========================================
    # CLUSTERING
    # ==========================================

    pivot_table['Total'] = pivot_table.sum(axis=1)
    filtered_data = pivot_table[pivot_table['Total'] > 1].drop(columns=['Total'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_data['Cluster'] = kmeans.fit_predict(scaled_data)

    st.subheader("Hasil Clustering")
    st.dataframe(filtered_data.head())

    # ==========================================
    # PILIH PRODUK
    # ==========================================

    produk = st.selectbox("Pilih Produk", filtered_data.index.tolist())

    data_produk = filtered_data.loc[produk].drop('Cluster', errors='ignore')
    data_produk = pd.to_numeric(data_produk)

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    st.line_chart(data_produk)

    metode = st.selectbox(
        "Pilih Metode",
        [
            "Holt-Winters Additive",
            "Holt-Winters Multiplicative",
            "ETS",
            "ARIMA",
            "Perbandingan Semua"
        ]
    )

    steps = st.slider("Forecast Bulan", 1, 12, 6)

    # ==========================================
    # 1. HOLT-WINTERS ADDITIVE
    # ==========================================

    if metode == "Holt-Winters Additive":

        model = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit = model.fit()
        forecast = fit.forecast(steps)

        st.line_chart(pd.concat([data_produk, forecast]))

    # ==========================================
    # 2. HOLT-WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":

        safe = data_produk.copy()
        safe[safe <= 0] = 1

        model = ExponentialSmoothing(
            safe,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        )

        fit = model.fit()
        forecast = fit.forecast(steps)

        st.line_chart(pd.concat([data_produk, forecast]))

    # ==========================================
    # 3. ETS
    # ==========================================

    elif metode == "ETS":

        model = ETSModel(
            data_produk,
            error='add',
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit = model.fit()
        forecast = fit.forecast(steps)

        st.line_chart(pd.concat([data_produk, forecast]))

    # ==========================================
    # 4. ARIMA
    # ==========================================

    elif metode == "ARIMA":

        model = ARIMA(data_produk, order=(1,1,1))
        fit = model.fit()

        forecast = fit.forecast(steps=steps)

        st.line_chart(pd.concat([data_produk, forecast]))

    # ==========================================
    # PERBANDINGAN
    # ==========================================

    else:

        # HW ADD
        model1 = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()

        # HW MUL
        safe = data_produk.copy()
        safe[safe <= 0] = 1

        model2 = ExponentialSmoothing(
            safe,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        ).fit()

        # ETS
        model3 = ETSModel(
            data_produk,
            error='add',
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()

        # ARIMA
        model4 = ARIMA(data_produk, order=(1,1,1)).fit()

        # FORECAST
        f1 = model1.forecast(steps)
        f2 = model2.forecast(steps)
        f3 = model3.forecast(steps)
        f4 = model4.forecast(steps)

        fig, ax = plt.subplots(figsize=(12,6))

        ax.plot(data_produk, label="Aktual")
        ax.plot(f1, label="HW Add")
        ax.plot(f2, label="HW Mul")
        ax.plot(f3, label="ETS")
        ax.plot(f4, label="ARIMA")

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # MAE (training)
        st.write("Perbandingan sederhana MAE")

        res = pd.DataFrame({
            "Metode": ["HW Add", "HW Mul", "ETS", "ARIMA"],
            "MAE": [
                mean_absolute_error(data_produk, model1.fittedvalues),
                mean_absolute_error(data_produk, model2.fittedvalues),
                mean_absolute_error(data_produk, model3.fittedvalues),
                mean_absolute_error(data_produk[1:], model4.predict(start=1, end=len(data_produk)-1))
            ]
        })

        st.dataframe(res.sort_values("MAE"))
