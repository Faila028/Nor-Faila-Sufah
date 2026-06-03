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
    # URUTAN BULAN — 36 BULAN (Jan-23 s/d Dec-25)
    # ==========================================

    urutan_bulan = [
        'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23',
        'May-23', 'Jun-23', 'Jul-23', 'Aug-23',
        'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
        'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24',
        'May-24', 'Jun-24', 'Jul-24', 'Aug-24',
        'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
        'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25',
        'May-25', 'Jun-25', 'Jul-25', 'Aug-25',
        'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25'
    ]

    pivot_table = pivot_table.reindex(
        columns=urutan_bulan,
        fill_value=0
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
    ].copy()

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
        figsize=(8, 5)
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
        figsize=(8, 5)
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

    data_produk = filtered_data.loc[produk].copy()

    kolom_hapus = ['Cluster', 'Kategori', 'Total']

    data_produk = data_produk.drop(kolom_hapus)

    data_produk = pd.to_numeric(data_produk)

    jumlah_bulan_aktif = (data_produk > 0).sum()

    if jumlah_bulan_aktif < 6:
        st.warning(
            "Produk ini hanya aktif kurang dari 6 bulan "
            "sehingga hasil forecasting kurang reliabel."
        )

    # ==========================================
    # INDEX TANGGAL — 36 BULAN
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
    # TRAIN-TEST SPLIT
    # Train: semua data kecuali n bulan terakhir
    # Test : n bulan terakhir (sesuai jumlah_forecast)
    # ==========================================

    n = len(data_produk)
    train = data_produk.iloc[:n - jumlah_forecast]
    test  = data_produk.iloc[n - jumlah_forecast:]

    if len(train) < 12:
        st.error(
            f"Data train hanya {len(train)} bulan. "
            "Kurangi jumlah forecast agar data train minimal 12 bulan."
        )
        st.stop()

    st.info(
        f"📌 Train: {len(train)} bulan  |  "
        f"Test: {len(test)} bulan  |  "
        f"MAE & RMSE dihitung dari hasil forecast vs data test."
    )

    # ==========================================
    # VISUAL DATA AKTUAL
    # ==========================================

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        data_produk.index,
        data_produk.values,
        marker='o',
        linewidth=2,
        label='Data Aktual'
    )

    ax2.axvline(
        x=test.index[0],
        color='red',
        linestyle='--',
        alpha=0.7,
        label='Awal Periode Test'
    )

    ax2.set_title(f'Data Aktual Produk {produk}')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig2)

    # ==========================================
    # FUNGSI TAMPILKAN HASIL
    # MAE & RMSE = forecast vs data test
    # ==========================================

    def tampilkan_hasil(nama_metode, forecast, test_actual):

        mae = mean_absolute_error(
            test_actual.values,
            forecast.values
        )

        rmse = np.sqrt(mean_squared_error(
            test_actual.values,
            forecast.values
        ))

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric("MAE", f"{mae:.2f}")

        with col_m2:
            st.metric("RMSE", f"{rmse:.2f}")

        forecast_df = pd.DataFrame({
            'Periode': forecast.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(forecast.values, 2),
            'Data Aktual (Test)': np.round(test_actual.values, 2)
        })

        forecast_df.index = range(1, len(forecast_df) + 1)

        st.subheader("📋 Hasil Forecast vs Data Aktual")

        st.dataframe(forecast_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            train.index,
            train.values,
            marker='o',
            label='Data Train'
        )

        ax.plot(
            test.index,
            test.values,
            marker='o',
            color='orange',
            label='Data Test (Aktual)'
        )

        ax.plot(
            forecast.index,
            forecast.values,
            marker='o',
            linestyle='--',
            color='green',
            label=f'Forecast {nama_metode}'
        )

        ax.set_title(f'Forecast {nama_metode} — Produk {produk}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)

        return mae, rmse

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

        forecast = fit.forecast(jumlah_forecast).clip(lower=0)

        tampilkan_hasil("HW Additive", forecast, test)

    # ==========================================
    # HOLT WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":

        train_nz = train.copy()
        train_nz[train_nz <= 0] = 1

        model = ExponentialSmoothing(
            train_nz,
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        )

        fit = model.fit()

        forecast = fit.forecast(jumlah_forecast).clip(lower=0)

        tampilkan_hasil("HW Multiplicative", forecast, test)

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":

        model = ETSModel(
            train,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )

        fit = model.fit(disp=False)

        forecast = fit.forecast(jumlah_forecast).clip(lower=0)

        tampilkan_hasil("ETS", forecast, test)

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":

        model = ARIMA(
            train,
            order=(1, 1, 1)
        )

        fit = model.fit()

        forecast = fit.forecast(steps=jumlah_forecast)

        tampilkan_hasil("ARIMA", forecast, test)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        hasil = {}

        # HW ADDITIVE
        fit_hw_add = ExponentialSmoothing(
            train, trend='add', seasonal='add', seasonal_periods=12
        ).fit()
        fc_hw_add = fit_hw_add.forecast(jumlah_forecast).clip(lower=0)
        hasil['HW Additive'] = {
            'forecast': fc_hw_add,
            'mae':  mean_absolute_error(test.values, fc_hw_add.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hw_add.values))
        }

        # HW MULTIPLICATIVE
        train_nz = train.copy()
        train_nz[train_nz <= 0] = 1
        fit_hw_mul = ExponentialSmoothing(
            train_nz, trend='add', seasonal='mul', seasonal_periods=12
        ).fit()
        fc_hw_mul = fit_hw_mul.forecast(jumlah_forecast).clip(lower=0)
        hasil['HW Multiplicative'] = {
            'forecast': fc_hw_mul,
            'mae':  mean_absolute_error(test.values, fc_hw_mul.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hw_mul.values))
        }

        # ETS
        fit_ets = ETSModel(
            train, error="add", trend="add", seasonal="add", seasonal_periods=12
        ).fit(disp=False)
        fc_ets = fit_ets.forecast(jumlah_forecast).clip(lower=0)
        hasil['ETS'] = {
            'forecast': fc_ets,
            'mae':  mean_absolute_error(test.values, fc_ets.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_ets.values))
        }

        # ARIMA
        fit_arima = ARIMA(train, order=(1, 1, 1)).fit()
        fc_arima = fit_arima.forecast(steps=jumlah_forecast)
        hasil['ARIMA'] = {
            'forecast': fc_arima,
            'mae':  mean_absolute_error(test.values, fc_arima.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_arima.values))
        }

        # ==========================================
        # TABEL PERBANDINGAN
        # ==========================================

        perbandingan = pd.DataFrame([
            {
                'Metode': m,
                'MAE':  round(v['mae'], 2),
                'RMSE': round(v['rmse'], 2)
            }
            for m, v in hasil.items()
        ])

        perbandingan.index = range(1, len(perbandingan) + 1)

        st.subheader("📊 Perbandingan Metode")

        st.dataframe(perbandingan, use_container_width=True)

        # ==========================================
        # GRAFIK MAE
        # ==========================================

        fig_mae, ax_mae = plt.subplots(figsize=(10, 5))

        ax_mae.bar(
            perbandingan['Metode'],
            perbandingan['MAE']
        )

        ax_mae.set_title('Perbandingan Nilai MAE')
        ax_mae.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig_mae)

        # ==========================================
        # METODE TERBAIK
        # ==========================================

        best_idx = perbandingan['MAE'].idxmin()
        metode_terbaik = perbandingan.loc[best_idx]

        st.success(
            f"✅ Metode terbaik adalah **{metode_terbaik['Metode']}** "
            f"dengan MAE = **{metode_terbaik['MAE']:.2f}** "
            f"dan RMSE = **{metode_terbaik['RMSE']:.2f}**"
        )

        # ==========================================
        # VISUALISASI GABUNGAN
        # ==========================================

        fig6, ax6 = plt.subplots(figsize=(14, 6))

        ax6.plot(
            train.index,
            train.values,
            marker='o',
            linewidth=2,
            color='steelblue',
            label='Data Train'
        )

        ax6.plot(
            test.index,
            test.values,
            marker='o',
            linewidth=2,
            color='orange',
            label='Data Test (Aktual)'
        )

        warna = ['green', 'red', 'purple', 'brown']

        for i, (nama, v) in enumerate(hasil.items()):
            ax6.plot(
                v['forecast'].index,
                v['forecast'].values,
                linestyle='--',
                marker='o',
                color=warna[i],
                label=nama
            )

        ax6.axvline(
            x=test.index[0],
            color='gray',
            linestyle=':',
            alpha=0.7,
            label='Awal Periode Test'
        )

        ax6.set_title(f'Perbandingan Forecast Produk {produk}')
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig6)
