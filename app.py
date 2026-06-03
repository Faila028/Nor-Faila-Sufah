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
        'Sep-24','Oct-24','Nov-24','Dec-24',

        'Jan-25','Feb-25','Mar-25','Apr-25',
        'May-25','Jun-25','Jul-25','Aug-25',
        'Sep-25','Oct-25','Nov-25','Dec-25'
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

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        filtered_data.drop(columns=['Total'])
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

    # ==========================================
    # MENENTUKAN FAST - MEDIUM - SLOW
    # ==========================================

    cluster_avg = filtered_data.groupby(
        'Cluster'
    )['Total'].mean().sort_values(
        ascending=False
    )

    mapping_cluster = {}

    if len(cluster_avg) >= 3:

        mapping_cluster[
            cluster_avg.index[0]
        ] = 'Fast Moving'

        mapping_cluster[
            cluster_avg.index[1]
        ] = 'Medium Moving'

        mapping_cluster[
            cluster_avg.index[2]
        ] = 'Slow Moving'

    filtered_data['Kategori'] = (
        filtered_data['Cluster']
        .map(mapping_cluster)
    )

    # ==========================================
    # JUMLAH PRODUK PER CLUSTER
    # ==========================================

    cluster_count = filtered_data[
        'Kategori'
    ].value_counts().reindex([
        'Fast Moving',
        'Medium Moving',
        'Slow Moving'
    ])

    tabel_cluster = pd.DataFrame({
        'Kategori': cluster_count.index,
        'Jumlah Produk': cluster_count.values
    })

    # INDEX DIMULAI DARI 1

    tabel_cluster.index = range(
        1,
        len(tabel_cluster) + 1
    )

    st.subheader(
        "📊 Jumlah Produk per Cluster"
    )

    st.dataframe(
        tabel_cluster,
        use_container_width=True
    )

    # ==========================================
    # VISUALISASI CLUSTER
    # ==========================================

    fig_cluster, ax_cluster = plt.subplots(
        figsize=(8,5)
    )

    ax_cluster.bar(
        tabel_cluster['Kategori'],
        tabel_cluster['Jumlah Produk']
    )

    ax_cluster.set_title(
        'Distribusi Produk per Cluster'
    )

    ax_cluster.set_xlabel(
        'Kategori Cluster'
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
        "Pilih Cluster",
        [
            'Fast Moving',
            'Medium Moving',
            'Slow Moving'
        ]
    )

    produk_cluster = filtered_data[
        filtered_data['Kategori']
        == pilih_cluster
    ].index.tolist()

    # ==========================================
    # TAMPILKAN PRODUK
    # ==========================================

    st.subheader(
        f"📦 Produk dalam {pilih_cluster}"
    )

    df_produk_cluster = pd.DataFrame({
        'Produk': produk_cluster
    })

    # INDEX DIMULAI DARI 1

    df_produk_cluster.index = range(
        1,
        len(df_produk_cluster) + 1
    )

    st.dataframe(
        df_produk_cluster,
        use_container_width=True
    )

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    daftar_produk = produk_cluster

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    data_produk = filtered_data.loc[
        produk
    ]

    kolom_hapus = [
        'Cluster',
        'Kategori',
        'Total'
    ]

    data_produk = data_produk.drop(
        kolom_hapus
    )

    data_produk = pd.to_numeric(
        data_produk
    )

    jumlah_bulan_aktif = (data_produk > 0).sum()

    if jumlah_bulan_aktif < 6:
        st.warning(
            "Produk ini hanya aktif kurang dari 6 bulan sehingga hasil forecasting kurang reliabel."
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
    # FUNGSI TAMPILKAN HASIL
    # ==========================================

    def tampilkan_hasil(
        nama_metode,
        forecast,
        test,
        mae,
        rmse
    ):

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "MAE",
                f"{mae:.2f}"
            )

        with col2:
            st.metric(
                "RMSE",
                f"{rmse:.2f}"
            )

        evaluasi = pd.DataFrame({
            "Aktual": test.values,
            "Forecast": np.round(
                forecast.values,
                2
            ),
            "Error": np.round(
                test.values - forecast.values,
                2
            ),
            "MAE Bulanan": np.round(
                np.abs(
                    test.values - forecast.values
                ),
                2
            )
        })

        evaluasi.index = test.index

        st.subheader(
            "📊 Evaluasi Forecast"
        )

        st.dataframe(
            evaluasi,
            use_container_width=True
        )

        forecast_df = pd.DataFrame({
            'Periode Forecast': forecast.index,
            'Hasil Forecast': np.round(
                forecast.values,
                2
            )
        })

        forecast_df.index = range(
            1,
            len(forecast_df)+1
        )

        st.subheader(
            "📋 Hasil Forecast"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )  

        fig, ax = plt.subplots(
            figsize=(12,5)
        )

        ax.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax.plot(
            forecast.index,
            forecast.values,
            marker='o',
            linestyle='--',
            linewidth=2,
            label=nama_metode
        )

        ax.axvline(
            test.index[0],
            color='red',
            linestyle='--',
            alpha=0.7
        )

        ax.legend()

        ax.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig)

    # ==========================================
    # HOLT WINTERS ADDITIVE
    # ==========================================

    if metode == "Holt-Winters Additive":

        train = data_produk[:-jumlah_forecast]

        test = data_produk[-jumlah_forecast:]

        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit = model.fit()

        forecast_test = fit.forecast(
            jumlah_forecast
        )

        forecast_test = forecast_test.clip(
            lower=0
        )

        # =====================
        # Forecast Masa Depan
        # =====================

        model_full = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit_full = model_full.fit()

        forecast_future = fit_full.forecast(
            jumlah_forecast
        )

        forecast_future = forecast_future.clip(
            lower=0
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

        tampilkan_hasil(
            "HW Additive",
            forecast,
            test,
            mae,
            rmse
        )

    # ==========================================
    # HOLT WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":

        train = data_produk[:-jumlah_forecast]

        test = data_produk[-jumlah_forecast:]

        train[train <= 0] = 1

        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
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

        tampilkan_hasil(
            "HW Multiplicative",
            forecast,
            test,
            mae,
            rmse
        )

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":

        train = data_produk[:-jumlah_forecast]

        test = data_produk[-jumlah_forecast:]

        model = ETSModel(
            train,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=12
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

        tampilkan_hasil(
            "ETS",
            forecast,
            test,
            mae,
            rmse
        )

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":

        train = data_produk[:-jumlah_forecast]

        test = data_produk[-jumlah_forecast:]

        model = ARIMA(
            train,
            order=(1,1,1)
        )

        fit = model.fit()

        forecast = fit.forecast(
            steps=jumlah_forecast
        )

        forecast.index = test.index

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

        tampilkan_hasil(
            "ARIMA",
            forecast,
            test,
            mae,
            rmse
        )

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        hasil = {}

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

        forecast_hw_add = forecast_hw_add.clip(
            lower=0
        )

        mae_hw_add = mean_absolute_error(
            data_produk,
            fit_hw_add.fittedvalues
        )

        # HW MULTIPLICATIVE

        data_nonzero = data_produk.copy()

        data_nonzero[
            data_nonzero <= 0
        ] = 1

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

        forecast_hw_mul = forecast_hw_mul.clip(
            lower=0
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

        forecast_ets = forecast_ets.clip(
            lower=0
        )

        mae_ets = mean_absolute_error(
            data_produk,
            fit_ets.fittedvalues
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
                'ARIMA'
            ],

            'MAE': [
                mae_hw_add,
                mae_hw_mul,
                mae_ets,
                mae_arima
            ]
        })

        perbandingan.index = range(
            1,
            len(perbandingan) + 1
        )

        st.subheader(
            "📊 Perbandingan Metode"
        )

        st.dataframe(
            perbandingan,
            use_container_width=True
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
