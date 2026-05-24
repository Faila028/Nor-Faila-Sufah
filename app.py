# app.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# CONFIG PAGE
# ==========================================
st.set_page_config(
    page_title="Clustering & Forecasting",
    layout="wide"
)

# ==========================================
# JUDUL
# ==========================================
st.title("Analisis Clustering dan Forecasting")
st.subheader("Metode K-Means dan Holt-Winters Multiplicative")

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

    # ==========================================
    # TAMPILKAN DATA AWAL
    # ==========================================
    st.write("## Data Awal")
    st.dataframe(df.head())

    # ==========================================
    # CEK KOLOM
    # ==========================================
    st.write("## Nama Kolom")
    st.write(df.columns)

    # ==========================================
    # PREPROCESSING
    # ==========================================

    # Ubah format tanggal
    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    # Hapus data keluar = 0
    df = df[df['keluar'] > 0]

    # Membuat format bulan
    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

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
    # URUTKAN BULAN
    # ==========================================
    urutan_bulan = [
        'Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23',
        'Jul-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24','May-24','Jun-24',
        'Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(columns=urutan_bulan)

    # ==========================================
    # TAMPILKAN PIVOT
    # ==========================================
    st.write("## Data Pivot Barang Keluar")
    st.dataframe(pivot_table)

    # ==========================================
    # DOWNLOAD PIVOT
    # ==========================================
    excel_data = pivot_table.to_csv().encode('utf-8')

    st.download_button(
        label="Download Pivot Table",
        data=excel_data,
        file_name='Data_Barang_Keluar_Perbulan.csv',
        mime='text/csv'
    )

    # ==========================================
    # CLUSTERING
    # ==========================================
    st.write("# Clustering K-Means")

    # Hapus kolom total jika ada
    if 'Total' in pivot_table.columns:
        pivot_table = pivot_table.drop(columns=['Total'])

    # Tambahkan total
    pivot_table['Total'] = pivot_table.sum(axis=1)

    # Filter produk
    filtered_data = pivot_table[pivot_table['Total'] > 1]

    # Hapus total lagi
    filtered_data = filtered_data.drop(columns=['Total'])

    # ==========================================
    # NORMALISASI
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

    # ==========================================
    # GRAFIK ELBOW
    # ==========================================
    st.write("## Grafik Elbow")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

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
    # K-MEANS
    # ==========================================
    kmeans = KMeans(
        n_clusters=jumlah_cluster,
        random_state=42
    )

    cluster = kmeans.fit_predict(scaled_data)

    # ==========================================
    # HASIL CLUSTER
    # ==========================================
    filtered_data['Cluster'] = cluster

    st.write("## Hasil Clustering")
    st.dataframe(filtered_data)

    # ==========================================
    # JUMLAH PRODUK TIAP CLUSTER
    # ==========================================
    st.write("## Jumlah Produk Tiap Cluster")

    cluster_count = filtered_data['Cluster'].value_counts()

    st.write(cluster_count)

    # ==========================================
    # GRAFIK CLUSTER
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    cluster_count.plot(kind='bar', ax=ax2)

    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Jumlah Produk')
    ax2.set_title('Jumlah Produk Tiap Cluster')

    st.pyplot(fig2)

    # ==========================================
    # RATA-RATA TOTAL PER CLUSTER
    # ==========================================
    filtered_data['Total'] = filtered_data.drop(columns=['Cluster']).sum(axis=1)

    cluster_summary = filtered_data.groupby('Cluster')['Total'].mean()

    st.write("## Rata-rata Total Barang Keluar")
    st.write(cluster_summary)

    # ==========================================
    # FAST MOVING
    # ==========================================
    st.write("# Fast Moving Product")

    cluster_fast = st.selectbox(
        "Pilih Cluster Fast Moving",
        filtered_data['Cluster'].unique()
    )

    fast_moving = filtered_data[
        filtered_data['Cluster'] == cluster_fast
    ]

    st.dataframe(fast_moving)

    st.write("## Daftar Produk")
    st.write(fast_moving.index.tolist())

    # ==========================================
    # FORECASTING
    # ==========================================
    st.write("# Forecasting Holt-Winters")

    # Pilih produk
    daftar_produk = filtered_data.index.tolist()

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # ==========================================
    # AMBIL DATA PRODUK
    # ==========================================
    data_produk = filtered_data.loc[produk]

    # Hapus cluster dan total
    data_produk = data_produk.drop(labels=['Cluster', 'Total'])

    # Ubah ke numeric
    data_produk = pd.to_numeric(data_produk)

    # ==========================================
    # TAMPILKAN DATA PRODUK
    # ==========================================
    st.write("## Data Produk")
    st.write(data_produk)

    # ==========================================
    # MODEL HOLT-WINTERS
    # ==========================================
    try:
        model = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()

        # Forecast 6 bulan
        forecast = model.forecast(6)

        # ==========================================
        # HASIL FORECAST
        # ==========================================
        st.write("## Hasil Forecast")
        st.write(forecast)

        # ==========================================
        # GRAFIK FORECAST
        # ==========================================
        fig3, ax3 = plt.subplots(figsize=(10, 5))

        data_produk.plot(label='Data Aktual', ax=ax3)
        forecast.plot(label='Forecast', ax=ax3)

        ax3.set_title(f'Forecast Produk {produk}')
        ax3.set_xlabel('Periode')
        ax3.set_ylabel('Jumlah Keluar')

        ax3.legend()

        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Forecast gagal dilakukan: {e}")

else:
    st.info("Silakan upload file Excel terlebih dahulu")
