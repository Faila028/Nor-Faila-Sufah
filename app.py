# app.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# CONFIG HALAMAN
# ==========================================

st.set_page_config(
    page_title="Forecasting Barang",
    layout="wide"
)

st.title("Forecasting dan Clustering Barang")

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
    # MEMBACA FILE EXCEL
    # ==========================================

    df = pd.read_excel(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df.head())

    # ==========================================
    # CEK NAMA KOLOM
    # ==========================================

    st.subheader("Nama Kolom")
    st.write(df.columns)

    # ==========================================
    # UBAH FORMAT TANGGAL
    # ==========================================

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    # ==========================================
    # HAPUS DATA KELUAR = 0
    # ==========================================

    df = df[df['keluar'] > 0]

    # ==========================================
    # MEMBUAT FORMAT BULAN
    # ==========================================

    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    # ==========================================
    # MEMBUAT PIVOT TABLE
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
        'Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23',
        'Jul-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24','May-24','Jun-24',
        'Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(columns=urutan_bulan)

    st.subheader("Pivot Table Barang Keluar")
    st.dataframe(pivot_table)

    # ==========================================
    # DOWNLOAD PIVOT TABLE
    # ==========================================

    excel_data = pivot_table.to_csv().encode('utf-8')

    st.download_button(
        label="Download Pivot Table",
        data=excel_data,
        file_name='data_barang_keluar.csv',
        mime='text/csv'
    )

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.header("Clustering Produk")

    if 'Total' in pivot_table.columns:
        pivot_table = pivot_table.drop(columns=['Total'])

    pivot_table['Total'] = pivot_table.sum(axis=1)

    filtered_data = pivot_table[pivot_table['Total'] > 1]

    filtered_data = filtered_data.drop(columns=['Total'])

    # ==========================================
    # NORMALISASI DATA
    # ==========================================

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data)

    # ==========================================
    # ELBOW METHOD
    # ==========================================

    inertia = []
    K = range(1, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots(figsize=(8,5))

    ax1.plot(K, inertia, marker='o')
    ax1.set_xlabel('Jumlah Cluster (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Metode Elbow')
    ax1.grid(True)

    st.pyplot(fig1)

    # ==========================================
    # PILIH JUMLAH CLUSTER
    # ==========================================

    jumlah_cluster = st.slider(
        "Pilih Jumlah Cluster",
        min_value=2,
        max_value=10,
        value=3
    )

    # ==========================================
    # K-MEANS CLUSTERING
    # ==========================================

    kmeans = KMeans(
        n_clusters=jumlah_cluster,
        random_state=42
    )

    cluster = kmeans.fit_predict(scaled_data)

    filtered_data['Cluster'] = cluster

    st.subheader("Hasil Clustering")
    st.dataframe(filtered_data.head())

    # ==========================================
    # JUMLAH PRODUK TIAP CLUSTER
    # ==========================================

    st.subheader("Jumlah Produk per Cluster")
    st.write(filtered_data['Cluster'].value_counts())

    # ==========================================
    # RATA-RATA TOTAL PER CLUSTER
    # ==========================================

    filtered_data['Total'] = filtered_data.drop(columns=['Cluster']).sum(axis=1)

    cluster_summary = filtered_data.groupby('Cluster')['Total'].mean()

    st.subheader("Rata-rata Total per Cluster")
    st.write(cluster_summary)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("Forecasting Holt-Winters")

    daftar_produk = filtered_data.index.tolist()

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # ==========================================
    # AMBIL DATA PRODUK
    # ==========================================

    data_produk = filtered_data.loc[produk]

    kolom_hapus = []

    if 'Cluster' in data_produk.index:
        kolom_hapus.append('Cluster')

    if 'Total' in data_produk.index:
        kolom_hapus.append('Total')

    data_produk = data_produk.drop(kolom_hapus)

    # ==========================================
    # GANTI 0 MENJADI 1
    # ==========================================

    data_produk = data_produk.replace(0, 1)

    # ==========================================
    # INDEX TANGGAL
    # ==========================================

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    # ==========================================
    # TAMPILKAN DATA AKTUAL
    # ==========================================

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(
        data_produk.index,
        data_produk.values,
        marker='o',
        label='Data Aktual'
    )

    ax2.set_title(f'Data Aktual Produk {produk}')
    ax2.set_xlabel('Periode')
    ax2.set_ylabel('Jumlah Barang Keluar')
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

    # ==========================================
    # MODEL HOLT WINTERS
    # ==========================================

    model = ExponentialSmoothing(
        data_produk,
        trend='add',
        seasonal='mul',
        seasonal_periods=12
    )

    fit_model = model.fit()

    # ==========================================
    # FORECAST
    # ==========================================

    jumlah_forecast = st.slider(
        "Jumlah Forecast Bulan",
        min_value=1,
        max_value=12,
        value=6
    )

    forecast = fit_model.forecast(jumlah_forecast)

    # Mengubah hasil negatif menjadi 0
    forecast = forecast.clip(lower=0)


    st.subheader("Hasil Forecast")
    st.write(forecast)

    # ==========================================
    # VISUALISASI FORECAST
    # ==========================================

    fig3, ax3 = plt.subplots(figsize=(12,5))

    ax3.plot(
        data_produk.index,
        data_produk.values,
        marker='o',
        label='Data Aktual'
    )

    ax3.plot(
        forecast.index,
        forecast.values,
        marker='o',
        linestyle='--',
        label='Forecast Holt-Winters'
    )

    ax3.set_title(f'Forecast Produk {produk}')
    ax3.set_xlabel('Periode')
    ax3.set_ylabel('Jumlah Barang Keluar')
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)

