# app.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =========================================
# CONFIG
# =========================================

st.set_page_config(
    page_title="Clustering dan Forecasting",
    layout="wide"
)

# =========================================
# TITLE
# =========================================

st.title("📦 Clustering dan Forecasting Permintaan Barang")
st.write("Analisis menggunakan metode K-Means dan Holt-Winters Multiplicative")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("Menu")
menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Upload Data",
        "Clustering",
        "Forecasting"
    ]
)

# =========================================
# UPLOAD FILE
# =========================================

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx"]
)

if uploaded_file is not None:

    # =========================================
    # READ DATA
    # =========================================

    df = pd.read_excel(uploaded_file)

    # =========================================
    # PREPROCESSING
    # =========================================

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    df = df[df['keluar'] > 0]

    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    # =========================================
    # SORT MONTHS
    # =========================================

    urutan_bulan = sorted(
        df['Bulan'].unique(),
        key=lambda x: pd.to_datetime(x, format='%b-%y')
    )

    # =========================================
    # PIVOT TABLE
    # =========================================

    pivot_table = df.pivot_table(
        index='id_produk',
        columns='Bulan',
        values='keluar',
        aggfunc='sum',
        fill_value=0
    )

    pivot_table = pivot_table.reindex(columns=urutan_bulan)

    # =========================================
    # TOTAL
    # =========================================

    pivot_table['Total'] = pivot_table.sum(axis=1)

    filtered_data = pivot_table[pivot_table['Total'] > 1].copy()

    # =========================================
    # CLUSTERING
    # =========================================

    fitur_cluster = filtered_data.drop(columns=['Total'])

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(fitur_cluster)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    filtered_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # =========================================
    # CLUSTER SUMMARY
    # =========================================

    cluster_summary = filtered_data.groupby('Cluster')['Total'].mean()

    fast_moving_cluster = cluster_summary.idxmax()

    filtered_data['Kategori'] = filtered_data['Cluster'].apply(
        lambda x: 'Fast Moving' if x == fast_moving_cluster else 'Slow/Medium Moving'
    )

    # =========================================
    # MENU : UPLOAD DATA
    # =========================================

    if menu == "Upload Data":

        st.subheader("Dataset")

        st.dataframe(df)

        st.subheader("Informasi Dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Jumlah Data", len(df))
        col2.metric("Jumlah Produk", df['id_produk'].nunique())
        col3.metric("Jumlah Bulan", len(urutan_bulan))

    # =========================================
    # MENU : CLUSTERING
    # =========================================

    elif menu == "Clustering":

        st.subheader("Hasil Clustering K-Means")

        st.dataframe(filtered_data)

        st.subheader("Rata-rata Total Barang Keluar")

        st.dataframe(cluster_summary.reset_index())

        # =========================================
        # VISUALISASI CLUSTER
        # =========================================

        fig, ax = plt.subplots(figsize=(8, 5))

        for cluster in filtered_data['Cluster'].unique():

            cluster_data = filtered_data[
                filtered_data['Cluster'] == cluster
            ]

            ax.scatter(
                cluster_data.index,
                cluster_data['Total'],
                label=f'Cluster {cluster}'
            )

        plt.xticks(rotation=90)

        ax.set_title('Visualisasi Cluster Produk')
        ax.set_xlabel('Produk')
        ax.set_ylabel('Total Barang Keluar')

        plt.legend()

        st.pyplot(fig)

        # =========================================
        # FAST MOVING
        # =========================================

        st.subheader("Produk Fast Moving")

        fast_moving = filtered_data[
            filtered_data['Kategori'] == 'Fast Moving'
        ]

        st.dataframe(fast_moving)

    # =========================================
    # MENU : FORECASTING
    # =========================================

    elif menu == "Forecasting":

        st.subheader("Forecasting Holt-Winters Multiplicative")

        produk_list = filtered_data.index.tolist()

        selected_produk = st.selectbox(
            "Pilih Produk",
            produk_list
        )

        # =========================================
        # DATA PRODUK
        # =========================================

        data_produk = pivot_table.loc[selected_produk]

        data_produk = data_produk.drop('Total')

        data_produk.index = pd.to_datetime(
            data_produk.index,
            format='%b-%y'
        )

        # =========================================
        # MODEL HOLT WINTERS
        # =========================================

        try:

            model = ExponentialSmoothing(
                data_produk,
                trend='add',
                seasonal='mul',
                seasonal_periods=12
            ).fit()

            forecast = model.forecast(6)

            # =========================================
            # TABEL FORECAST
            # =========================================

            forecast_df = pd.DataFrame({
                'Periode': forecast.index.strftime('%b-%Y'),
                'Forecast': forecast.values
            })

            st.subheader("Hasil Forecasting")

            st.dataframe(forecast_df)

            # =========================================
            # GRAFIK
            # =========================================

            fig2, ax2 = plt.subplots(figsize=(10, 5))

            ax2.plot(
                data_produk.index,
                data_produk.values,
                label='Data Aktual'
            )

            ax2.plot(
                forecast.index,
                forecast.values,
                label='Forecast'
            )

            ax2.set_title(
                f'Forecasting Produk {selected_produk}'
            )

            ax2.set_xlabel('Periode')
            ax2.set_ylabel('Jumlah Keluar')

            plt.legend()

            st.pyplot(fig2)

        except:

            st.error(
                "Data tidak cukup untuk Holt-Winters Multiplicative. Pastikan data minimal 2 musim (24 bulan)."
            )

else:

    st.info("Silakan upload file Excel terlebih dahulu")
```

---
