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

    total_produk = filtered_data['Total']

    filtered_data_cluster = filtered_data.drop(
        columns=['Total']
    )

    # ==========================================
    # NORMALISASI
    # ==========================================

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        filtered_data_cluster
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
    # JUMLAH CLUSTER
    # ==========================================

    jumlah_cluster = 3

    st.info(
        "Jumlah cluster ditetapkan sebanyak 3 kategori:\n"
        "- Fast Moving\n"
        "- Medium Moving\n"
        "- Slow Moving"
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

    filtered_data_cluster['Cluster_Asli'] = cluster

    # ==========================================
    # RANKING CLUSTER
    # ==========================================

    filtered_data_cluster['Total'] = total_produk.values

    rata_cluster = filtered_data_cluster.groupby(
        'Cluster_Asli'
    )['Total'].mean().sort_values(
        ascending=False
    )

    mapping_cluster = {}

    kategori_cluster = [
        'Fast Moving',
        'Medium Moving',
        'Slow Moving'
    ]

    for i, cluster_id in enumerate(rata_cluster.index):

        mapping_cluster[cluster_id] = kategori_cluster[i]

    filtered_data_cluster['Kategori'] = (
        filtered_data_cluster['Cluster_Asli']
        .map(mapping_cluster)
    )

    # ==========================================
    # TABEL CLUSTER
    # ==========================================

    st.subheader(
        "📊 Jumlah Produk per Cluster"
    )

    cluster_count = filtered_data_cluster[
        'Kategori'
    ].value_counts().reindex(
        kategori_cluster
    )

    df_cluster = pd.DataFrame({
        'Kategori': cluster_count.index,
        'Jumlah Produk': cluster_count.values
    })

    df_cluster.index = np.arange(
        1,
        len(df_cluster) + 1
    )

    st.dataframe(df_cluster)

    # ==========================================
    # GRAFIK CLUSTER
    # ==========================================

    fig_cluster, ax_cluster = plt.subplots(
        figsize=(8,5)
    )

    ax_cluster.bar(
        cluster_count.index,
        cluster_count.values
    )

    ax_cluster.set_title(
        'Distribusi Produk per Cluster'
    )

    ax_cluster.set_ylabel(
        'Jumlah Produk'
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
        "Pilih Kategori Cluster",
        kategori_cluster
    )

    produk_cluster = filtered_data_cluster[
        filtered_data_cluster['Kategori']
        == pilih_cluster
    ].index.tolist()

    st.subheader(
        f"📦 Produk dalam {pilih_cluster}"
    )

    df_produk = pd.DataFrame({
        'Produk': produk_cluster
    })

    df_produk.index = np.arange(
        1,
        len(df_produk) + 1
    )

    st.dataframe(df_produk)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    daftar_produk = (
        filtered_data_cluster.index.tolist()
    )

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # ==========================================
    # DATA PRODUK
    # ==========================================

    data_produk = filtered_data_cluster.loc[
        produk
    ]

    kolom_hapus = [
        'Cluster_Asli',
        'Kategori',
        'Total'
    ]

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
            "ARIMA",
            "Perbandingan Semua Metode"
        ]
    )

    # ==========================================
    # FORECAST
    # ==========================================

    jumlah_forecast = st.slider(
        "Jumlah Forecast Bulan",
        1,
        6,
        3
    )

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    train = data_produk[:-jumlah_forecast]

    test = data_produk[-jumlah_forecast:]

    # ==========================================
    # VALIDASI DATA
    # ==========================================

    if len(train) < 12:

        st.warning(
            "Data training terlalu sedikit "
            "untuk forecasting musiman."
        )

        st.stop()

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
    # HOLT WINTERS ADDITIVE
    # ==========================================

    if metode == "Holt-Winters Additive":

        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=6
        )

        fit = model.fit()

        forecast = fit.forecast(
            jumlah_forecast
        )

        forecast = forecast.clip(
            lower=0
        )

        mae = mean_absolute_error(
            test,
            forecast
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                forecast
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", f"{mae:.2f}")

        with col2:
            st.metric("RMSE", f"{rmse:.2f}")

        hasil_forecast = pd.DataFrame({
            'Periode': forecast.index,
            'Forecast': forecast.values
        })

        hasil_forecast.index = np.arange(
            1,
            len(hasil_forecast) + 1
        )

        st.subheader("📋 Hasil Forecast")

        st.dataframe(hasil_forecast)

    # ==========================================
    # HOLT WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":

        train_nonzero = train.copy()

        train_nonzero[
            train_nonzero <= 0
        ] = 1

        model = ExponentialSmoothing(
            train_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=6
        )

        fit = model.fit()

        forecast = fit.forecast(
            jumlah_forecast
        )

        forecast = forecast.clip(
            lower=0
        )

        mae = mean_absolute_error(
            test,
            forecast
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                forecast
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", f"{mae:.2f}")

        with col2:
            st.metric("RMSE", f"{rmse:.2f}")

        hasil_forecast = pd.DataFrame({
            'Periode': forecast.index,
            'Forecast': forecast.values
        })

        hasil_forecast.index = np.arange(
            1,
            len(hasil_forecast) + 1
        )

        st.subheader("📋 Hasil Forecast")

        st.dataframe(hasil_forecast)

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":

        model = ETSModel(
            train,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=6
        )

        fit = model.fit()

        forecast = fit.forecast(
            jumlah_forecast
        )

        forecast = forecast.clip(
            lower=0
        )

        mae = mean_absolute_error(
            test,
            forecast
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                forecast
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", f"{mae:.2f}")

        with col2:
            st.metric("RMSE", f"{rmse:.2f}")

        hasil_forecast = pd.DataFrame({
            'Periode': forecast.index,
            'Forecast': forecast.values
        })

        hasil_forecast.index = np.arange(
            1,
            len(hasil_forecast) + 1
        )

        st.subheader("📋 Hasil Forecast")

        st.dataframe(hasil_forecast)

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":

        model = ARIMA(
            train,
            order=(1,1,1)
        )

        fit = model.fit()

        forecast = fit.forecast(
            steps=jumlah_forecast
        )

        forecast = forecast.clip(
            lower=0
        )

        mae = mean_absolute_error(
            test,
            forecast
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                forecast
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", f"{mae:.2f}")

        with col2:
            st.metric("RMSE", f"{rmse:.2f}")

        hasil_forecast = pd.DataFrame({
            'Periode': forecast.index,
            'Forecast': forecast.values
        })

        hasil_forecast.index = np.arange(
            1,
            len(hasil_forecast) + 1
        )

        st.subheader("📋 Hasil Forecast")

        st.dataframe(hasil_forecast)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    elif metode == "Perbandingan Semua Metode":

        # HW ADD

        model_hw_add = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=6
        )

        fit_hw_add = model_hw_add.fit()

        forecast_hw_add = fit_hw_add.forecast(
            jumlah_forecast
        )

        forecast_hw_add = forecast_hw_add.clip(
            lower=0
        )

        mae_hw_add = mean_absolute_error(
            test,
            forecast_hw_add
        )

        # HW MULTI

        train_nonzero = train.copy()

        train_nonzero[
            train_nonzero <= 0
        ] = 1

        model_hw_mul = ExponentialSmoothing(
            train_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=6
        )

        fit_hw_mul = model_hw_mul.fit()

        forecast_hw_mul = fit_hw_mul.forecast(
            jumlah_forecast
        )

        forecast_hw_mul = forecast_hw_mul.clip(
            lower=0
        )

        mae_hw_mul = mean_absolute_error(
            test,
            forecast_hw_mul
        )

        # ETS

        model_ets = ETSModel(
            train,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=6
        )

        fit_ets = model_ets.fit()

        forecast_ets = fit_ets.forecast(
            jumlah_forecast
        )

        forecast_ets = forecast_ets.clip(
            lower=0
        )

        mae_ets = mean_absolute_error(
            test,
            forecast_ets
        )

        # ARIMA

        model_arima = ARIMA(
            train,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.forecast(
            steps=jumlah_forecast
        )

        forecast_arima = forecast_arima.clip(
            lower=0
        )

        mae_arima = mean_absolute_error(
            test,
            forecast_arima
        )

        # ==========================================
        # TABEL PERBANDINGAN
        # ==========================================

        st.subheader(
            "📊 Perbandingan Metode Forecasting"
        )

        perbandingan = pd.DataFrame({

             'Metode': [
                'HW Additive',
                'HW Multiplicative',
                   'ETS',
                'ARIMA'
            ],

            'MAE': [
                mae_hw_add,
                mae_hw_mul,
                mae_ets,
                mae_arima
            ]
        })

        # ==========================================
        # INDEX MULAI DARI 1
        # ==========================================

        perbandingan.index = np.arange(
            1,
            len(perbandingan) + 1
        )

        # ==========================================
        # CARI MAE TERKECIL
        # ==========================================

        min_mae = perbandingan['MAE'].min()

        # ==========================================
        # FUNGSI WARNA
        # ==========================================

        def highlight_best(row):

            if row['MAE'] == min_mae:

                return [
                    'background-color: lightgreen',
                    'background-color: lightgreen'
                ]

            else:

                return ['', '']

        # ==========================================
        # TAMPILKAN TABEL
        # ==========================================

        styled_df = perbandingan.style.apply(
            highlight_best,
            axis=1
        )

        st.dataframe(styled_df)

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
        # GRAFIK PERBANDINGAN
        # ==========================================

        fig_all, ax_all = plt.subplots(
            figsize=(14,6)
        )

        ax_all.plot(
            train.index,
            train.values,
            marker='o',
            linewidth=2,
            label='Training'
        )

        ax_all.plot(
            test.index,
            test.values,
            marker='o',
            linewidth=2,
            label='Data Aktual'
        )

        ax_all.plot(
            forecast_hw_add.index,
            forecast_hw_add.values,
            linestyle='--',
            marker='o',
            label='HW Additive'
        )

        ax_all.plot(
            forecast_hw_mul.index,
            forecast_hw_mul.values,
            linestyle='--',
            marker='o',
            label='HW Multiplicative'
        )

        ax_all.plot(
            forecast_ets.index,
            forecast_ets.values,
            linestyle='--',
            marker='o',
            label='ETS'
        )

        ax_all.plot(
            forecast_arima.index,
            forecast_arima.values,
            linestyle='--',
            marker='o',
            label='ARIMA'
        )

        ax_all.set_title(
            f'Perbandingan Forecast Produk {produk}'
        )

        ax_all.legend()

        ax_all.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig_all)
