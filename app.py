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

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

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
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stDownloadButton>button {
    background-color: #28a745;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

[data-testid="stMetricValue"] {
    color: #1f77b4;
    font-size: 28px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# JUDUL
# ==========================================

st.title("📦 Clustering dan Forecasting Permintaan Barang")

st.markdown("""
### Sistem Analisis Barang Keluar

Aplikasi ini digunakan untuk:
- Clustering produk menggunakan K-Means
- Forecasting permintaan barang
- Perbandingan beberapa metode forecasting
""")

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=["xlsx"]
)

# ==========================================
# JIKA FILE SUDAH DIUPLOAD
# ==========================================

if uploaded_file is not None:

    # ==========================================
    # MEMBACA FILE
    # ==========================================

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Awal")

    st.dataframe(df.head())

    # ==========================================
    # DASHBOARD
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Data",
            len(df)
        )

    with col2:
        st.metric(
            "Jumlah Produk",
            df['id_produk'].nunique()
        )

    with col3:
        st.metric(
            "Total Barang Keluar",
            int(df['keluar'].sum())
        )

    # ==========================================
    # FORMAT TANGGAL
    # ==========================================

    df['tgl_input'] = pd.to_datetime(
        df['tgl_input']
    )

    # ==========================================
    # FORMAT BULAN
    # ==========================================

    df['Bulan'] = df['tgl_input'].dt.strftime(
        '%b-%y'
    )

    # ==========================================
    # PIVOT TABLE
    # ==========================================

    pivot_table = df.pivot_table(
        index='id_produk',
        columns='Bulan',
        values='keluar',
        aggfunc='sum',
        fill_value=0
    )

    # ==========================================
    # URUTAN BULAN
    # ==========================================

    urutan_bulan = [
        'Jan-23','Feb-23','Mar-23','Apr-23',
        'May-23','Jun-23','Jul-23','Aug-23',
        'Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24',
        'May-24','Jun-24','Jul-24','Aug-24',
        'Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(
        columns=urutan_bulan
    )

    st.subheader(
        "📊 Pivot Table Barang Keluar"
    )

    st.dataframe(pivot_table)

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================

    csv_data = pivot_table.to_csv().encode(
        'utf-8'
    )

    st.download_button(
        label="⬇️ Download Pivot Table",
        data=csv_data,
        file_name='data_barang_keluar.csv',
        mime='text/csv'
    )

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.header("📌 Clustering Produk")

    pivot_table['Total'] = pivot_table.sum(
        axis=1
    )

    filtered_data = pivot_table[
        pivot_table['Total'] > 1
    ]

    filtered_data = filtered_data.drop(
        columns=['Total']
    )

    # ==========================================
    # NORMALISASI
    # ==========================================

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        filtered_data
    )

    # ==========================================
    # ELBOW METHOD
    # ==========================================

    inertia = []

    K = range(1, 10)

    for k in K:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42
        )

        kmeans.fit(scaled_data)

        inertia.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots(
        figsize=(8,5)
    )

    ax1.plot(
        K,
        inertia,
        marker='o'
    )

    ax1.set_title(
        'Metode Elbow'
    )

    ax1.set_xlabel(
        'Jumlah Cluster'
    )

    ax1.set_ylabel(
        'Inertia'
    )

    ax1.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    st.pyplot(fig1)

    # ==========================================
    # PILIH CLUSTER
    # ==========================================

    jumlah_cluster = st.slider(
        "Pilih Jumlah Cluster",
        min_value=2,
        max_value=10,
        value=3
    )

    # ==========================================
    # K-MEANS
    # ==========================================

    kmeans = KMeans(
        n_clusters=jumlah_cluster,
        random_state=42
    )

    cluster = kmeans.fit_predict(
        scaled_data
    )

    filtered_data['Cluster'] = cluster

    st.subheader(
        "📌 Hasil Clustering"
    )

    st.dataframe(
        filtered_data.head()
    )

    # ==========================================
    # JUMLAH PRODUK
    # ==========================================

    st.subheader(
        "📊 Jumlah Produk per Cluster"
    )

    cluster_count = filtered_data[
        'Cluster'
    ].value_counts()

    st.write(cluster_count)

    # ==========================================
    # GRAFIK CLUSTER
    # ==========================================

    fig_cluster, ax_cluster = plt.subplots(
        figsize=(7,5)
    )

    ax_cluster.bar(
        cluster_count.index.astype(str),
        cluster_count.values
    )

    ax_cluster.set_title(
        'Distribusi Produk per Cluster'
    )

    ax_cluster.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    st.pyplot(fig_cluster)

    # ==========================================
    # PILIH CLUSTER
    # ==========================================

    pilih_cluster = st.selectbox(
        "Pilih Cluster",
        sorted(
            filtered_data['Cluster'].unique()
        )
    )

    produk_cluster = filtered_data[
        filtered_data['Cluster'] == pilih_cluster
    ].index.tolist()

    st.subheader(
        f"📦 Produk dalam Cluster {pilih_cluster}"
    )

    st.dataframe(
        pd.DataFrame({
            'Produk': produk_cluster
        })
    )

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    daftar_produk = filtered_data.index.tolist()

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # ==========================================
    # DATA PRODUK
    # ==========================================

    data_produk = filtered_data.loc[produk]

    kolom_hapus = []

    if 'Cluster' in data_produk.index:
        kolom_hapus.append('Cluster')

    data_produk = data_produk.drop(
        kolom_hapus
    )

    data_produk = pd.to_numeric(
        data_produk
    )

    # ==========================================
    # INDEX TANGGAL
    # ==========================================

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    # ==========================================
    # PILIH METODE
    # ==========================================

    metode = st.selectbox(
        "Pilih Metode Forecasting",
        [
            "Holt-Winters Additive",
            "Holt-Winters Multiplicative",
            "ETS",
            "Damped Trend",
            "ARIMA",
            "Perbandingan Semua Metode"
        ]
    )

    jumlah_forecast = st.slider(
        "Jumlah Forecast Bulan",
        1,
        12,
        6
    )

    # ==========================================
    # VISUAL DATA AKTUAL
    # ==========================================

    fig2, ax2 = plt.subplots(
        figsize=(12,5)
    )

    ax2.plot(
        data_produk.index,
        data_produk.values,
        marker='o',
        linewidth=2,
        label='Data Aktual'
    )

    ax2.set_title(
        f'Data Aktual Produk {produk}'
    )

    ax2.legend()

    ax2.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    st.pyplot(fig2)

    # ==========================================
    # HOLT-WINTERS ADDITIVE
    # ==========================================

    if metode == "Holt-Winters Additive":

        model_hw_add = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit_hw_add = model_hw_add.fit()

        forecast_hw_add = fit_hw_add.forecast(
            jumlah_forecast
        )

        forecast_hw_add = forecast_hw_add.clip(
            lower=0
        )

        mae_hw_add = mean_absolute_error(
            data_produk,
            fit_hw_add.fittedvalues
        )

        rmse_hw_add = np.sqrt(
            mean_squared_error(
                data_produk,
                fit_hw_add.fittedvalues
            )
        )

        st.metric(
            "MAE",
            f"{mae_hw_add:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse_hw_add:.2f}"
        )

        st.write(forecast_hw_add)

        fig3, ax3 = plt.subplots(
            figsize=(12,5)
        )

        ax3.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax3.plot(
            forecast_hw_add.index,
            forecast_hw_add.values,
            marker='o',
            linestyle='--',
            label='HW Additive'
        )

        ax3.legend()

        ax3.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig3)

    # ==========================================
    # HOLT-WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":

        data_nonzero = data_produk.copy()

        data_nonzero[data_nonzero <= 0] = 1

        model_hw_mul = ExponentialSmoothing(
            data_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        )

        fit_hw_mul = model_hw_mul.fit()

        forecast_hw_mul = fit_hw_mul.forecast(
            jumlah_forecast
        )

        mae_hw_mul = mean_absolute_error(
            data_nonzero,
            fit_hw_mul.fittedvalues
        )

        rmse_hw_mul = np.sqrt(
            mean_squared_error(
                data_nonzero,
                fit_hw_mul.fittedvalues
            )
        )

        st.metric(
            "MAE",
            f"{mae_hw_mul:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse_hw_mul:.2f}"
        )

        st.write(forecast_hw_mul)

        fig_mul, ax_mul = plt.subplots(
            figsize=(12,5)
        )

        ax_mul.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax_mul.plot(
            forecast_hw_mul.index,
            forecast_hw_mul.values,
            marker='o',
            linestyle='--',
            label='HW Multiplicative'
        )

        ax_mul.legend()

        ax_mul.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig_mul)

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":

        model_ets = ETSModel(
            data_produk,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )

        fit_ets = model_ets.fit()

        forecast_ets = fit_ets.forecast(
            jumlah_forecast
        )

        mae_ets = mean_absolute_error(
            data_produk,
            fit_ets.fittedvalues
        )

        rmse_ets = np.sqrt(
            mean_squared_error(
                data_produk,
                fit_ets.fittedvalues
            )
        )

        st.metric(
            "MAE",
            f"{mae_ets:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse_ets:.2f}"
        )

        st.write(forecast_ets)

        fig_ets, ax_ets = plt.subplots(
            figsize=(12,5)
        )

        ax_ets.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax_ets.plot(
            forecast_ets.index,
            forecast_ets.values,
            marker='o',
            linestyle='--',
            label='ETS'
        )

        ax_ets.legend()

        ax_ets.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig_ets)

    # ==========================================
    # DAMPED TREND
    # ==========================================

    elif metode == "Damped Trend":

        model_damped = ExponentialSmoothing(
            data_produk,
            trend='add',
            damped_trend=True
        )

        fit_damped = model_damped.fit()

        forecast_damped = fit_damped.forecast(
            jumlah_forecast
        )

        mae_damped = mean_absolute_error(
            data_produk,
            fit_damped.fittedvalues
        )

        rmse_damped = np.sqrt(
            mean_squared_error(
                data_produk,
                fit_damped.fittedvalues
            )
        )

        st.metric(
            "MAE",
            f"{mae_damped:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse_damped:.2f}"
        )

        st.write(forecast_damped)

        fig_damped, ax_damped = plt.subplots(
            figsize=(12,5)
        )

        ax_damped.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax_damped.plot(
            forecast_damped.index,
            forecast_damped.values,
            marker='o',
            linestyle='--',
            label='Damped Trend'
        )

        ax_damped.legend()

        ax_damped.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig_damped)

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":

        model_arima = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.forecast(
            steps=jumlah_forecast
        )

        fitted_arima = fit_arima.predict(
            start=1,
            end=len(data_produk)-1
        )

        actual_arima = data_produk[1:]

        mae_arima = mean_absolute_error(
            actual_arima,
            fitted_arima
        )

        rmse_arima = np.sqrt(
            mean_squared_error(
                actual_arima,
                fitted_arima
            )
        )

        st.metric(
            "MAE",
            f"{mae_arima:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse_arima:.2f}"
        )

        st.write(forecast_arima)

        fig_arima, ax_arima = plt.subplots(
            figsize=(12,5)
        )

        ax_arima.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax_arima.plot(
            forecast_arima.index,
            forecast_arima.values,
            marker='o',
            linestyle='--',
            label='ARIMA'
        )

        ax_arima.legend()

        ax_arima.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig_arima)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        # HW ADDITIVE

        model_hw_add = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit_hw_add = model_hw_add.fit()

        forecast_hw_add = fit_hw_add.forecast(
            jumlah_forecast
        )

        mae_hw_add = mean_absolute_error(
            data_produk,
            fit_hw_add.fittedvalues
        )

        # HW MULTIPLICATIVE

        data_nonzero = data_produk.copy()

        data_nonzero[data_nonzero <= 0] = 1

        model_hw_mul = ExponentialSmoothing(
            data_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        )

        fit_hw_mul = model_hw_mul.fit()

        forecast_hw_mul = fit_hw_mul.forecast(
            jumlah_forecast
        )

        mae_hw_mul = mean_absolute_error(
            data_nonzero,
            fit_hw_mul.fittedvalues
        )

        # ETS

        model_ets = ETSModel(
            data_produk,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )

        fit_ets = model_ets.fit()

        forecast_ets = fit_ets.forecast(
            jumlah_forecast
        )

        mae_ets = mean_absolute_error(
            data_produk,
            fit_ets.fittedvalues
        )

        # DAMPED

        model_damped = ExponentialSmoothing(
            data_produk,
            trend='add',
            damped_trend=True
        )

        fit_damped = model_damped.fit()

        forecast_damped = fit_damped.forecast(
            jumlah_forecast
        )

        mae_damped = mean_absolute_error(
            data_produk,
            fit_damped.fittedvalues
        )

        # ARIMA

        model_arima = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.forecast(
            steps=jumlah_forecast
        )

        fitted_arima = fit_arima.predict(
            start=1,
            end=len(data_produk)-1
        )

        actual_arima = data_produk[1:]

        mae_arima = mean_absolute_error(
            actual_arima,
            fitted_arima
        )

        # ==========================================
        # TABEL PERBANDINGAN
        # ==========================================

        perbandingan = pd.DataFrame({

            'Metode': [
                'HW Additive',
                'HW Multiplicative',
                'ETS',
                'Damped Trend',
                'ARIMA'
            ],

            'MAE': [
                mae_hw_add,
                mae_hw_mul,
                mae_ets,
                mae_damped,
                mae_arima
            ]
        })

        st.subheader(
            "📊 Perbandingan Metode"
        )

        st.dataframe(
            perbandingan.style.highlight_min(
                axis=0,
                color='lightgreen'
            )
        )

        # ==========================================
        # GRAFIK MAE
        # ==========================================

        fig_mae, ax_mae = plt.subplots(
            figsize=(10,5)
        )

        ax_mae.bar(
            perbandingan['Metode'],
            perbandingan['MAE']
        )

        ax_mae.set_title(
            'Perbandingan Nilai MAE'
        )

        ax_mae.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig_mae)

        # ==========================================
        # METODE TERBAIK
        # ==========================================

        metode_terbaik = perbandingan.loc[
            perbandingan['MAE'].idxmin()
        ]

        st.success(
            f"Metode terbaik adalah "
            f"{metode_terbaik['Metode']} "
            f"dengan nilai MAE "
            f"{metode_terbaik['MAE']:.2f}"
        )

        # ==========================================
        # VISUALISASI GABUNGAN
        # ==========================================

        fig6, ax6 = plt.subplots(
            figsize=(14,6)
        )

        ax6.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            linewidth=2,
            label='Data Aktual'
        )

        ax6.plot(
            forecast_hw_add.index,
            forecast_hw_add.values,
            linestyle='--',
            marker='o',
            label='HW Additive'
        )

        ax6.plot(
            forecast_hw_mul.index,
            forecast_hw_mul.values,
            linestyle='--',
            marker='o',
            label='HW Multiplicative'
        )

        ax6.plot(
            forecast_ets.index,
            forecast_ets.values,
            linestyle='--',
            marker='o',
            label='ETS'
        )

        ax6.plot(
            forecast_damped.index,
            forecast_damped.values,
            linestyle='--',
            marker='o',
            label='Damped Trend'
        )

        ax6.plot(
            forecast_arima.index,
            forecast_arima.values,
            linestyle='--',
            marker='o',
            label='ARIMA'
        )

        ax6.set_title(
            f'Perbandingan Forecast Produk {produk}'
        )

        ax6.legend()

        ax6.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig6)
