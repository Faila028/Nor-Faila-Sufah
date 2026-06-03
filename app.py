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

    jumlah_bulan_aktif = (
        data_produk > 0
    ).sum()

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
    # JUMLAH FORECAST
    # ==========================================

    jumlah_forecast = st.slider(
        "Jumlah Forecast Bulan",
        1,
        6,
        3
    )

    if len(data_produk) <= jumlah_forecast + 6:

        st.warning(
            "Data terlalu sedikit untuk evaluasi forecasting."
        )

        st.stop()

    train = data_produk[
        :-jumlah_forecast
    ]

    test = data_produk[
        -jumlah_forecast:
    ]

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
        pred_test,
        forecast_future,
        mae,
        rmse
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "MAE Test",
                f"{mae:.2f}"
            )

        with col2:

            st.metric(
                "RMSE Test",
                f"{rmse:.2f}"
            )

        forecast_df = pd.DataFrame({

            "Periode Forecast":
            forecast_future.index.strftime(
                "%b-%Y"
            ),

            "Forecast":
            np.round(
                forecast_future.values,
                2
            )
        })

        forecast_df.index = range(
            1,
            len(forecast_df)+1
        )

        st.subheader(
            "📋 Forecast Bulan Mendatang"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(14,6)
        )

        ax.plot(
            train.index,
            train.values,
            marker='o',
            linewidth=2,
            label='Data Train'
        )

        ax.plot(
            test.index,
            test.values,
            marker='o',
            linewidth=2,
            label='Data Test'
        )

        ax.plot(
            pred_test.index,
            pred_test.values,
            marker='o',
            linestyle='--',
            linewidth=2,
            label='Test Forecast'
        )

        ax.plot(
            forecast_future.index,
            forecast_future.values,
            marker='o',
            linestyle='--',
            linewidth=2,
            label='Forecast Masa Depan'
        )

        ax.set_title(
            f'{nama_metode} - {produk}'
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

        seasonal = "add"
        seasonal_periods = 6

        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )

        fit = model.fit()

        pred_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            pred_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                pred_test
            )
        )

        model_full = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )

        fit_full = model_full.fit()

        forecast_future = fit_full.forecast(
            jumlah_forecast
        )

        tampilkan_hasil(
            "HW Additive",
            pred_test,
            forecast_future,
            mae,
            rmse
        )

    # ==========================================
    # HOLT WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":

        data_nonzero = train.copy()

        data_nonzero[
            data_nonzero <= 0
        ] = 1

        model = ExponentialSmoothing(
            data_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=6
        )

        fit = model.fit()

        pred_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            pred_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                pred_test
            )
        )

        full_nonzero = data_produk.copy()

        full_nonzero[
            full_nonzero <= 0
        ] = 1

        model_full = ExponentialSmoothing(
            full_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=6
        )

        fit_full = model_full.fit()

        forecast_future = fit_full.forecast(
            jumlah_forecast
        )

        forecast_future = forecast_future.clip(
            lower=0
        )

        tampilkan_hasil(
            "HW Multiplicative",
            pred_test,
            forecast_future,
            mae,
            rmse
        )

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":

        seasonal = (
            "add"
            if len(train) >= 24
            else None
        )

        model = ETSModel(
            train,
            error="add",
            trend="add",
            seasonal=seasonal,
            seasonal_periods=6 if seasonal else None
        )

        fit = model.fit()

        pred_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            pred_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                pred_test
            )
        )

        model_full = ETSModel(
            data_produk,
            error="add",
            trend="add",
            seasonal=seasonal,
            seasonal_periods=6 if seasonal else None
        )

        fit_full = model_full.fit()

        forecast_future = fit_full.forecast(
            jumlah_forecast
        )

        forecast_future = forecast_future.clip(
            lower=0
        )

        tampilkan_hasil(
            "ETS",
            pred_test,
            forecast_future,
            mae,
            rmse
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

        pred_test = fit.forecast(
            len(test)
        )

        mae = mean_absolute_error(
            test,
            pred_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                pred_test
            )
        )

        model_full = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_full = model_full.fit()

        forecast_future = fit_full.forecast(
            steps=jumlah_forecast
        )

        forecast_future = forecast_future.clip(
            lower=0
        )

        tampilkan_hasil(
            "ARIMA",
            pred_test,
            forecast_future,
            mae,
            rmse
        )

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        hasil = []

        # ==========================================
        # HW ADDITIVE
        # ==========================================

        model_hw_add = ExponentialSmoothing(
            train,
            trend='add',
            seasonal=None
        )

        fit_hw_add = model_hw_add.fit()

        pred_hw_add = fit_hw_add.forecast(
            len(test)
        )

        mae_hw_add = mean_absolute_error(
            test,
            pred_hw_add
        )

        rmse_hw_add = np.sqrt(
            mean_squared_error(
                test,
                pred_hw_add
            )
        )

        hasil.append([
            "HW Additive",
            mae_hw_add,
            rmse_hw_add
        ])

        # ==========================================
        # HW MULTIPLICATIVE
        # ==========================================

        train_nonzero = train.copy()

        train_nonzero[
            train_nonzero <= 0
        ] = 1

        try:

            model_hw_mul = ExponentialSmoothing(
                train_nonzero,
                trend='add',
                seasonal='mul',
                seasonal_periods=6
            )

            fit_hw_mul = model_hw_mul.fit()

            pred_hw_mul = fit_hw_mul.forecast(
                len(test)
            )

            mae_hw_mul = mean_absolute_error(
                test,
                pred_hw_mul
            )

            rmse_hw_mul = np.sqrt(
                mean_squared_error(
                    test,
                    pred_hw_mul
                )
            )

            hasil.append([
                "HW Multiplicative",
                mae_hw_mul,
                rmse_hw_mul
            ])

        except:

            pass

        # ==========================================
        # ETS
        # ==========================================

        model_ets = ETSModel(
            train,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=6
        )

        fit_ets = model_ets.fit()

        pred_ets = fit_ets.forecast(
            len(test)
        )

        mae_ets = mean_absolute_error(
            test,
            pred_ets
        )

        rmse_ets = np.sqrt(
            mean_squared_error(
                test,
                pred_ets
            )
        )

        hasil.append([
            "ETS",
            mae_ets,
            rmse_ets
        ])

        # ==========================================
        # ARIMA
        # ==========================================

        model_arima = ARIMA(
            train,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        pred_arima = fit_arima.forecast(
            len(test)
        )

        mae_arima = mean_absolute_error(
            test,
            pred_arima
        )

        rmse_arima = np.sqrt(
            mean_squared_error(
                test,
                pred_arima
            )
        )

        hasil.append([
            "ARIMA",
            mae_arima,
            rmse_arima
        ])

        # ==========================================
        # TABEL PERBANDINGAN
        # ==========================================

        perbandingan = pd.DataFrame(
            hasil,
            columns=[
                "Metode",
                "MAE",
                "RMSE"
            ]
        )

        perbandingan = perbandingan.sort_values(
            by="MAE"
        )

        perbandingan.index = range(
            1,
            len(perbandingan) + 1
        )

        st.subheader(
            "📊 Perbandingan Semua Metode"
        )

        st.dataframe(
            perbandingan,
            use_container_width=True
        )

        metode_terbaik = perbandingan.iloc[0]

        st.success(
            f"Metode terbaik adalah "
            f"{metode_terbaik['Metode']} "
            f"dengan MAE "
            f"{metode_terbaik['MAE']:.2f}"
        )

        # ==========================================
        # GRAFIK MAE
        # ==========================================

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        ax.bar(
            perbandingan["Metode"],
            perbandingan["MAE"]
        )

        ax.set_title(
            "Perbandingan Nilai MAE"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.5
        )

        st.pyplot(fig)

        # ==========================================
        # FORECAST MASING-MASING METODE
        # ==========================================

        model_hw_add_full = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=6
        )

        fit_hw_add_full = model_hw_add_full.fit()

        forecast_hw_add = fit_hw_add_full.forecast(
            jumlah_forecast
        )

        full_nonzero = data_produk.copy()

        full_nonzero[
            full_nonzero <= 0
        ] = 1

        model_hw_mul = ExponentialSmoothing(
           train_nonzero,
            trend='add',
            seasonal='mul',
            seasonal_periods=6
        )

        fit_hw_mul_full = model_hw_mul_full.fit()

        forecast_hw_mul = fit_hw_mul_full.forecast(
            jumlah_forecast
        )

        model_ets_full = ETSModel(
            data_produk,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=6
        )

        fit_ets_full = model_ets_full.fit()

        forecast_ets = fit_ets_full.forecast(
            jumlah_forecast
        )

        model_arima_full = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_arima_full = model_arima_full.fit()

        forecast_arima = fit_arima_full.forecast(
            steps=jumlah_forecast
        )

        # ==========================================
        # GRAFIK PERBANDINGAN FORECAST
        # ==========================================

        st.subheader(
            "📈 Perbandingan Forecast Semua Metode"
        )

        fig2, ax2 = plt.subplots(
            figsize=(14,6)
        )

        ax2.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            linewidth=2,
            label='Data Aktual'
        )

        ax2.plot(
            forecast_hw_add.index,
            forecast_hw_add.values,
            '--o',
            label='HW Additive'
        )

        ax2.plot(
            forecast_hw_mul.index,
            forecast_hw_mul.values,
            '--o',
            label='HW Multiplicative'
        )

        ax2.plot(
            forecast_ets.index,
            forecast_ets.values,
            '--o',
            label='ETS'
        )

        ax2.plot(
            forecast_arima.index,
            forecast_arima.values,
            '--o',
            label='ARIMA'
        )

        ax2.set_title(
            f'Perbandingan Forecast Produk {produk}'
        )

        ax2.legend()

        ax2.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        st.pyplot(fig2)
