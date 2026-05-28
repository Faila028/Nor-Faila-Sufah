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

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing
)

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

st.title("📦 Clustering dan Forecasting Permintaan Barang")

uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=["xlsx"]
)

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
        'Jan-23','Feb-23','Mar-23','Apr-23',
        'May-23','Jun-23','Jul-23','Aug-23',
        'Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24',
        'May-24','Jun-24','Jul-24','Aug-24',
        'Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(columns=urutan_bulan)

    st.subheader("Pivot Table")
    st.dataframe(pivot_table)

    # ================= CLUSTERING =================

    pivot_table['Total'] = pivot_table.sum(axis=1)

    filtered_data = pivot_table[pivot_table['Total'] > 1].drop(columns=['Total'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_data['Cluster'] = kmeans.fit_predict(scaled_data)

    st.subheader("Hasil Clustering")
    st.dataframe(filtered_data.head())

    # ================= FORECASTING =================

    produk = st.selectbox("Pilih Produk", filtered_data.index.tolist())

    data_produk = filtered_data.loc[produk].drop('Cluster', errors='ignore')
    data_produk = pd.to_numeric(data_produk)

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    metode = st.selectbox(
        "Pilih Metode Forecasting",
        [
            "Holt-Winters Additive",
            "Holt-Winters Multiplicative",
            "ETS",
            "ARIMA",
            "Perbandingan Semua Metode"
        ]
    )

    jumlah_forecast = st.slider("Jumlah Forecast", 1, 12, 6)

    # ================= DATA AKTUAL =================

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data_produk.index, data_produk.values, marker='o', label='Aktual')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # ================= HOLT WINTERS ADD =================

    if metode == "Holt-Winters Additive":

        model = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit = model.fit()
        forecast = fit.forecast(jumlah_forecast).clip(lower=0)

        mae = mean_absolute_error(data_produk, fit.fittedvalues)
        rmse = np.sqrt(mean_squared_error(data_produk, fit.fittedvalues))

        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data_produk.index, data_produk.values, label='Aktual')
        ax.plot(forecast.index, forecast.values, '--', label='Forecast')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    # ================= HOLT WINTERS MULTI =================

    elif metode == "Holt-Winters Multiplicative":

        data_nonzero = data_produk.copy()
        data_nonzero[data_nonzero <= 0] = 1

        model = ExponentialSmoothing(
            data_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        )

        fit = model.fit()
        forecast = fit.forecast(jumlah_forecast)

        mae = mean_absolute_error(data_nonzero, fit.fittedvalues)
        rmse = np.sqrt(mean_squared_error(data_nonzero, fit.fittedvalues))

        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data_produk.index, data_produk.values, label='Aktual')
        ax.plot(forecast.index, forecast.values, '--', label='Forecast')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    # ================= ETS =================

    elif metode == "ETS":

        model = ETSModel(
            data_produk,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )

        fit = model.fit()
        forecast = fit.forecast(jumlah_forecast)

        mae = mean_absolute_error(data_produk, fit.fittedvalues)
        rmse = np.sqrt(mean_squared_error(data_produk, fit.fittedvalues))

        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data_produk.index, data_produk.values, label='Aktual')
        ax.plot(forecast.index, forecast.values, '--', label='Forecast ETS')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    # ================= ARIMA =================

    elif metode == "ARIMA":

        model = ARIMA(data_produk, order=(1,1,1))
        fit = model.fit()

        forecast = fit.forecast(steps=jumlah_forecast)

        fitted = fit.predict(start=1, end=len(data_produk)-1)
        actual = data_produk[1:]

        mae = mean_absolute_error(actual, fitted)
        rmse = np.sqrt(mean_squared_error(actual, fitted))

        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data_produk.index, data_produk.values, label='Aktual')
        ax.plot(forecast.index, forecast.values, '--', label='Forecast ARIMA')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    # ================= PERBANDINGAN =================

    else:

        # HW ADD
        model1 = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        fit1 = model1.fit()

        # HW MULTI
        data_nonzero = data_produk.copy()
        data_nonzero[data_nonzero <= 0] = 1

        model2 = ExponentialSmoothing(
            data_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        )
        fit2 = model2.fit()

        # ETS
        model3 = ETSModel(
            data_produk,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )
        fit3 = model3.fit()

        # ARIMA
        model4 = ARIMA(data_produk, order=(1,1,1))
        fit4 = model4.fit()

        # MAE
        df_compare = pd.DataFrame({
            "Metode": ["HW Add", "HW Multi", "ETS", "ARIMA"],
            "MAE": [
                mean_absolute_error(data_produk, fit1.fittedvalues),
                mean_absolute_error(data_nonzero, fit2.fittedvalues),
                mean_absolute_error(data_produk, fit3.fittedvalues),
                mean_absolute_error(data[1:], fit4.predict(start=1, end=len(data_produk)-1))
            ]
        })

        st.dataframe(df_compare)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(df_compare["Metode"], df_compare["MAE"])
        ax.grid()
        st.pyplot(fig)

        best = df_compare.loc[df_compare["MAE"].idxmin()]

        st.success(f"Metode terbaik: {best['Metode']} (MAE {best['MAE']:.2f})")
