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
# FUNCTION MAPE
# ==========================================

def mape(actual, pred):

    actual = np.array(actual)
    pred = np.array(pred)

    actual = np.where(actual == 0, 1, actual)

    return np.mean(
        np.abs((actual - pred) / actual)
    ) * 100

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=["xlsx"]
)

# ==========================================
# JIKA FILE ADA
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

    # ==========================================
    # TOTAL PENJUALAN
    # ==========================================

    total_penjualan = filtered_data['Total']

    clustering_data = filtered_data.drop(
        columns=['Total']
    )

    # ==========================================
    # NORMALISASI
    # ==========================================

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        clustering_data
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
        marker='o',
        linewidth=2
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

    clustering_data['Cluster'] = cluster

    clustering_data['Total'] = total_penjualan

    # ==========================================
    # URUTKAN CLUSTER
    # ==========================================

    rata_cluster = clustering_data.groupby(
        'Cluster'
    )['Total'].mean().sort_values(
        ascending=False
    )

    mapping_cluster = {}

    for i, cluster_lama in enumerate(
        rata_cluster.index
    ):

        mapping_cluster[cluster_lama] = i + 1

    clustering_data['Cluster'] = clustering_data[
        'Cluster'
    ].map(mapping_cluster)

    # ==========================================
    # LABEL CLUSTER
    # ==========================================

    nama_cluster = {
        1: 'Fast Moving',
        2: 'Medium Moving',
        3: 'Slow Moving'
    }

    clustering_data['Kategori'] = clustering_data[
        'Cluster'
    ].map(nama_cluster)

    # ==========================================
    # HASIL CLUSTER
    # ==========================================

    st.subheader(
        "📌 Hasil Clustering"
    )

    st.dataframe(
        clustering_data.head()
    )

    # ==========================================
    # JUMLAH PRODUK PER CLUSTER
    # ==========================================

    st.subheader(
        "📊 Jumlah Produk per Cluster"
    )

    cluster_count = clustering_data.groupby(
        'Kategori'
    ).size().reset_index(
        name='Jumlah Produk'
    )

    st.dataframe(cluster_count)

    # ==========================================
    # VISUALISASI CLUSTER
    # ==========================================

    fig_cluster, ax_cluster = plt.subplots(
        figsize=(8,5)
    )

    ax_cluster.bar(
        cluster_count['Kategori'],
        cluster_count['Jumlah Produk']
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
        sorted(
            clustering_data['Cluster'].unique()
        )
    )

    produk_cluster = clustering_data[
        clustering_data['Cluster'] == pilih_cluster
    ].index.tolist()

    nama_kategori = nama_cluster[
        pilih_cluster
    ]

    st.subheader(
        f"📦 Produk dalam {nama_kategori}"
    )

    df_produk = pd.DataFrame({
        'Produk': produk_cluster
    })

    df_produk.index = range(
        1,
        len(df_produk) + 1
    )

    st.dataframe(df_produk)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    daftar_produk = clustering_data.index.tolist()

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # ==========================================
    # DATA PRODUK
    # ==========================================

    data_produk = clustering_data.loc[produk]

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

    # ==========================================
    # INDEX TANGGAL
    # ==========================================

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    train_size = int(
        len(data_produk) * 0.8
    )

    train = data_produk.iloc[:train_size]

    test = data_produk.iloc[train_size:]

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
    # HOLT WINTERS ADDITIVE
    # ==========================================

    if metode == "Holt-Winters Additive":

        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit = model.fit()

        prediksi_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            prediksi_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                prediksi_test
            )
        )

        nilai_mape = mape(
            test,
            prediksi_test
        )

        forecast = fit.forecast(
            jumlah_forecast
        )

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
            seasonal_periods=12
        )

        fit = model.fit()

        prediksi_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            prediksi_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                prediksi_test
            )
        )

        nilai_mape = mape(
            test,
            prediksi_test
        )

        forecast = fit.forecast(
            jumlah_forecast
        )

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":

        model = ETSModel(
            train,
            error='add',
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit = model.fit()

        prediksi_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            prediksi_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                prediksi_test
            )
        )

        nilai_mape = mape(
            test,
            prediksi_test
        )

        forecast = fit.forecast(
            jumlah_forecast
        )

    # ==========================================
    # DAMPED TREND
    # ==========================================

    elif metode == "Damped Trend":

        model = ExponentialSmoothing(
            train,
            trend='add',
            damped_trend=True
        )

        fit = model.fit()

        prediksi_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            prediksi_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                prediksi_test
            )
        )

        nilai_mape = mape(
            test,
            prediksi_test
        )

        forecast = fit.forecast(
            jumlah_forecast
        )

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":

        model = ARIMA(
            train,
            order=(1,1,1)
        )

        fit = model.fit()

        prediksi_test = fit.forecast(
            steps=len(test)
        )

        mae = mean_absolute_error(
            test,
            prediksi_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                prediksi_test
            )
        )

        nilai_mape = mape(
            test,
            prediksi_test
        )

        forecast = fit.forecast(
            steps=jumlah_forecast
        )

    # ==========================================
    # HASIL METRIK
    # ==========================================

    if metode != "Perbandingan Semua Metode":

        col1, col2, col3 = st.columns(3)

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

        with col3:
            st.metric(
                "MAPE",
                f"{nilai_mape:.2f}%"
            )

        # ==========================================
        # TABEL FORECAST
        # ==========================================

        forecast_df = pd.DataFrame({

            'Tanggal Forecast': forecast.index.strftime(
                '%b-%Y'
            ),

            'Jumlah Prediksi Barang': np.round(
                forecast.values,
                2
            )
        })

        forecast_df.index = range(
            1,
            len(forecast_df) + 1
        )

        st.subheader(
            "📋 Hasil Forecast"
        )

        st.dataframe(
            forecast_df
        )

        # ==========================================
        # VISUAL FORECAST
        # ==========================================

        fig3, ax3 = plt.subplots(
            figsize=(13,6)
        )

        ax3.plot(
            train.index,
            train.values,
            marker='o',
            linewidth=2,
            label='Training'
        )

        ax3.plot(
            test.index,
            test.values,
            marker='o',
            linewidth=2,
            label='Testing'
        )

        ax3.plot(
            forecast.index,
            forecast.values,
            marker='o',
            linestyle='--',
            linewidth=2,
            label='Forecast'
        )

        ax3.set_title(
            f'Forecast Produk {produk}'
        )

        ax3.legend()

        ax3.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig3)
