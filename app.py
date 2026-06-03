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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
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
.main { background-color: #f5f7fa; }
h1 { color: #1f4e79; text-align: center; font-weight: bold; }
h2, h3 { color: #1f4e79; }
.stButton>button {
    background-color: #1f77b4; color: white;
    border-radius: 10px; border: none;
    padding: 10px 20px; font-weight: bold;
}
.stDownloadButton>button {
    background-color: #28a745; color: white;
    border-radius: 10px; border: none;
    padding: 10px 20px; font-weight: bold;
}
[data-testid="stMetricValue"] { color: #1f77b4; font-size: 28px; }
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

uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

# ==========================================
# JIKA FILE SUDAH DIUPLOAD
# ==========================================

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Awal")
    st.dataframe(df.head())

    # DASHBOARD
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", len(df))
    with col2:
        st.metric("Jumlah Produk", df['id_produk'].nunique())
    with col3:
        st.metric("Total Barang Keluar", int(df['keluar'].sum()))

    # FORMAT TANGGAL & BULAN
    df['tgl_input'] = pd.to_datetime(df['tgl_input'])
    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    # PIVOT TABLE
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

    pivot_table = pivot_table.reindex(columns=urutan_bulan, fill_value=0)

    st.subheader("📊 Pivot Table Barang Keluar")
    st.dataframe(pivot_table)

    csv_data = pivot_table.to_csv().encode('utf-8')
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

    pivot_table['Total'] = pivot_table.sum(axis=1)
    filtered_data = pivot_table[pivot_table['Total'] > 1].copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data.drop(columns=['Total']))

    # ELBOW METHOD
    inertia = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertia.append(km.inertia_)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(K, inertia, marker='o')
    ax1.set_title('Metode Elbow')
    ax1.set_xlabel('Jumlah Cluster')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    jumlah_cluster = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42)
    cluster = kmeans.fit_predict(scaled_data)
    filtered_data['Cluster'] = cluster

    # FAST - MEDIUM - SLOW
    cluster_avg = filtered_data.groupby('Cluster')['Total'].mean().sort_values(ascending=False)
    mapping_cluster = {}
    if len(cluster_avg) >= 3:
        mapping_cluster[cluster_avg.index[0]] = 'Fast Moving'
        mapping_cluster[cluster_avg.index[1]] = 'Medium Moving'
        mapping_cluster[cluster_avg.index[2]] = 'Slow Moving'

    filtered_data['Kategori'] = filtered_data['Cluster'].map(mapping_cluster)

    cluster_count = filtered_data['Kategori'].value_counts().reindex(
        ['Fast Moving', 'Medium Moving', 'Slow Moving']
    )
    tabel_cluster = pd.DataFrame({
        'Kategori': cluster_count.index,
        'Jumlah Produk': cluster_count.values
    })
    tabel_cluster.index = range(1, len(tabel_cluster) + 1)

    st.subheader("📊 Jumlah Produk per Cluster")
    st.dataframe(tabel_cluster, use_container_width=True)

    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
    ax_cluster.bar(tabel_cluster['Kategori'], tabel_cluster['Jumlah Produk'])
    ax_cluster.set_title('Distribusi Produk per Cluster')
    ax_cluster.set_xlabel('Kategori Cluster')
    ax_cluster.set_ylabel('Jumlah Produk')
    ax_cluster.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_cluster)

    pilih_cluster = st.selectbox(
        "Pilih Cluster",
        ['Fast Moving', 'Medium Moving', 'Slow Moving']
    )

    produk_cluster = filtered_data[
        filtered_data['Kategori'] == pilih_cluster
    ].index.tolist()

    st.subheader(f"📦 Produk dalam {pilih_cluster}")
    df_produk_cluster = pd.DataFrame({'Produk': produk_cluster})
    df_produk_cluster.index = range(1, len(df_produk_cluster) + 1)
    st.dataframe(df_produk_cluster, use_container_width=True)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    produk = st.selectbox("Pilih Produk", produk_cluster)

    data_produk = filtered_data.loc[produk].copy()
    data_produk = data_produk.drop(['Cluster', 'Kategori', 'Total'])
    data_produk = pd.to_numeric(data_produk)

    jumlah_bulan_aktif = (data_produk > 0).sum()
    if jumlah_bulan_aktif < 6:
        st.warning("Produk ini hanya aktif kurang dari 6 bulan sehingga hasil forecasting kurang reliabel.")

    # INDEX TANGGAL — 36 BULAN
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

    jumlah_forecast = st.slider("Jumlah Forecast Bulan", 1, 12, 6)

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
        f"📌 Train: {len(train)} bulan  |  Test: {len(test)} bulan  |  "
        "MAE & RMSE dihitung dari forecast vs data test."
    )

    # GRAFIK DATA AKTUAL
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(data_produk.index, data_produk.values, marker='o', linewidth=2, label='Data Aktual')
    ax2.axvline(x=test.index[0], color='red', linestyle='--', alpha=0.7, label='Awal Periode Test')
    ax2.set_title(f'Data Aktual Produk {produk}')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    # ==========================================
    # HELPER: RETRAIN DENGAN FULL DATA
    # ==========================================

    def retrain_full(nama_metode, full_data, jumlah_forecast):
        """Melatih ulang model dengan seluruh data lalu forecast ke depan."""
        if nama_metode == 'HW Additive':
            model = ExponentialSmoothing(
                full_data, trend='add', seasonal='add', seasonal_periods=12
            ).fit()
            return model.forecast(jumlah_forecast).clip(lower=0)
        elif nama_metode == 'HW Multiplicative':
            full_nz = full_data.copy()
            full_nz[full_nz <= 0] = 1
            model = ExponentialSmoothing(
                full_nz, trend='add', seasonal='mul', seasonal_periods=12
            ).fit()
            return model.forecast(jumlah_forecast).clip(lower=0)
        elif nama_metode == 'ETS':
            model = ETSModel(
                full_data, error="add", trend="add", seasonal="add", seasonal_periods=12
            ).fit(disp=False)
            return model.forecast(jumlah_forecast).clip(lower=0)
        elif nama_metode == 'ARIMA':
            model = ARIMA(full_data, order=(1, 1, 1)).fit()
            return model.forecast(steps=jumlah_forecast)

    # Palet warna konsisten
    warna_metode = {
        'HW Additive':       'green',
        'HW Multiplicative': 'red',
        'ETS':               'purple',
        'ARIMA':             'brown'
    }

    # ==========================================
    # FUNGSI TAMPILKAN HASIL (1 METODE)
    # ==========================================

    def tampilkan_hasil(nama_metode, forecast_eval, test_actual):

        # --- BAGIAN 1: EVALUASI ---
        mae  = mean_absolute_error(test_actual.values, forecast_eval.values)
        rmse = np.sqrt(mean_squared_error(test_actual.values, forecast_eval.values))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("MAE", f"{mae:.2f}")
        with col_m2:
            st.metric("RMSE", f"{rmse:.2f}")

        eval_df = pd.DataFrame({
            'Periode': forecast_eval.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(forecast_eval.values, 2),
            'Data Aktual (Test)': np.round(test_actual.values, 2)
        })
        eval_df.index = range(1, len(eval_df) + 1)
        st.subheader("📋 Hasil Forecast vs Data Aktual (Evaluasi)")
        st.dataframe(eval_df, use_container_width=True)

        fig_eval, ax_eval = plt.subplots(figsize=(12, 5))
        ax_eval.plot(train.index, train.values, marker='o', color='steelblue', label='Data Train')
        ax_eval.plot(test_actual.index, test_actual.values, marker='o', color='orange', label='Data Test (Aktual)')
        ax_eval.plot(
            forecast_eval.index, forecast_eval.values,
            marker='o', linestyle='--',
            color=warna_metode.get(nama_metode, 'green'),
            label=f'Forecast {nama_metode}'
        )
        ax_eval.axvline(x=test_actual.index[0], color='gray', linestyle=':', alpha=0.7, label='Awal Periode Test')
        ax_eval.set_title(f'Evaluasi Model — {nama_metode} — Produk {produk}')
        ax_eval.legend()
        ax_eval.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_eval)

        # --- BAGIAN 2: FORECAST KE DEPAN ---
        st.subheader("🔮 Forecast ke Depan")
        st.info(
            f"Model di-retrain menggunakan seluruh {len(data_produk)} bulan data, "
            "lalu meramalkan bulan-bulan berikutnya."
        )

        future_forecast = retrain_full(nama_metode, data_produk, jumlah_forecast)

        future_df = pd.DataFrame({
            'Periode': future_forecast.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(future_forecast.values, 2)
        })
        future_df.index = range(1, len(future_df) + 1)
        st.dataframe(future_df, use_container_width=True)

        # Download
        csv_future = future_df.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Forecast ke Depan",
            data=csv_future,
            file_name=f'forecast_ke_depan_{produk}_{nama_metode}.csv',
            mime='text/csv'
        )

        fig_future, ax_future = plt.subplots(figsize=(12, 5))
        ax_future.plot(data_produk.index, data_produk.values, marker='o', linewidth=2, color='steelblue', label=f'Data Aktual ({len(data_produk)} bulan)')
        ax_future.plot(
            future_forecast.index, future_forecast.values,
            marker='o', linestyle='--',
            color=warna_metode.get(nama_metode, 'green'),
            linewidth=2,
            label=f'Forecast ke Depan ({nama_metode})'
        )
        ax_future.axvline(x=future_forecast.index[0], color='red', linestyle='--', alpha=0.7, label='Awal Periode Forecast')
        ax_future.set_title(f'Forecast ke Depan — {nama_metode} — Produk {produk}')
        ax_future.legend()
        ax_future.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_future)

        return mae, rmse

    # ==========================================
    # HOLT-WINTERS ADDITIVE
    # ==========================================

    if metode == "Holt-Winters Additive":
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(jumlah_forecast).clip(lower=0)
        tampilkan_hasil("HW Additive", forecast, test)

    # ==========================================
    # HOLT-WINTERS MULTIPLICATIVE
    # ==========================================

    elif metode == "Holt-Winters Multiplicative":
        train_nz = train.copy()
        train_nz[train_nz <= 0] = 1
        model = ExponentialSmoothing(train_nz, trend='add', seasonal='mul', seasonal_periods=12).fit()
        forecast = model.forecast(jumlah_forecast).clip(lower=0)
        tampilkan_hasil("HW Multiplicative", forecast, test)

    # ==========================================
    # ETS
    # ==========================================

    elif metode == "ETS":
        model = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12).fit(disp=False)
        forecast = model.forecast(jumlah_forecast).clip(lower=0)
        tampilkan_hasil("ETS", forecast, test)

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":
        model = ARIMA(train, order=(1, 1, 1)).fit()
        forecast = model.forecast(steps=jumlah_forecast)
        tampilkan_hasil("ARIMA", forecast, test)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        hasil_eval   = {}
        hasil_future = {}

        # HW ADDITIVE
        fit_hwa = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        fc_hwa  = fit_hwa.forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Additive'] = {
            'forecast': fc_hwa,
            'mae':  mean_absolute_error(test.values, fc_hwa.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwa.values))
        }
        hasil_future['HW Additive'] = retrain_full('HW Additive', data_produk, jumlah_forecast)

        # HW MULTIPLICATIVE
        train_nz = train.copy()
        train_nz[train_nz <= 0] = 1
        fit_hwm = ExponentialSmoothing(train_nz, trend='add', seasonal='mul', seasonal_periods=12).fit()
        fc_hwm  = fit_hwm.forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Multiplicative'] = {
            'forecast': fc_hwm,
            'mae':  mean_absolute_error(test.values, fc_hwm.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwm.values))
        }
        hasil_future['HW Multiplicative'] = retrain_full('HW Multiplicative', data_produk, jumlah_forecast)

        # ETS
        fit_ets = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12).fit(disp=False)
        fc_ets  = fit_ets.forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['ETS'] = {
            'forecast': fc_ets,
            'mae':  mean_absolute_error(test.values, fc_ets.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_ets.values))
        }
        hasil_future['ETS'] = retrain_full('ETS', data_produk, jumlah_forecast)

        # ARIMA
        fit_arima = ARIMA(train, order=(1, 1, 1)).fit()
        fc_arima  = fit_arima.forecast(steps=jumlah_forecast)
        hasil_eval['ARIMA'] = {
            'forecast': fc_arima,
            'mae':  mean_absolute_error(test.values, fc_arima.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_arima.values))
        }
        hasil_future['ARIMA'] = retrain_full('ARIMA', data_produk, jumlah_forecast)

        # ==========================================
        # TABEL PERBANDINGAN MAE & RMSE
        # ==========================================

        perbandingan = pd.DataFrame([
            {'Metode': m, 'MAE': round(v['mae'], 2), 'RMSE': round(v['rmse'], 2)}
            for m, v in hasil_eval.items()
        ])
        perbandingan.index = range(1, len(perbandingan) + 1)

        st.subheader("📊 Perbandingan MAE & RMSE Semua Metode")
        st.dataframe(perbandingan, use_container_width=True)

        # Grafik MAE
        fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
        bars = ax_mae.bar(
            perbandingan['Metode'], perbandingan['MAE'],
            color=[warna_metode[m] for m in perbandingan['Metode']]
        )
        ax_mae.bar_label(bars, fmt='%.2f', padding=3)
        ax_mae.set_title('Perbandingan Nilai MAE — Semua Metode')
        ax_mae.set_ylabel('MAE')
        ax_mae.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_mae)

        # Grafik RMSE
        fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
        bars2 = ax_rmse.bar(
            perbandingan['Metode'], perbandingan['RMSE'],
            color=[warna_metode[m] for m in perbandingan['Metode']]
        )
        ax_rmse.bar_label(bars2, fmt='%.2f', padding=3)
        ax_rmse.set_title('Perbandingan Nilai RMSE — Semua Metode')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_rmse)

        # Metode terbaik
        best_idx       = perbandingan['MAE'].idxmin()
        metode_terbaik = perbandingan.loc[best_idx]
        st.success(
            f"✅ Metode terbaik adalah **{metode_terbaik['Metode']}** "
            f"dengan MAE = **{metode_terbaik['MAE']:.2f}** "
            f"dan RMSE = **{metode_terbaik['RMSE']:.2f}**"
        )

        # ==========================================
        # GRAFIK EVALUASI GABUNGAN
        # Train | Test Aktual | Forecast tiap metode
        # ==========================================

        st.subheader("📉 Grafik Evaluasi Gabungan (Train vs Test vs Forecast)")

        fig_eval, ax_eval = plt.subplots(figsize=(14, 6))
        ax_eval.plot(train.index, train.values, marker='o', linewidth=2, color='steelblue', label='Data Train')
        ax_eval.plot(test.index, test.values, marker='o', linewidth=2, color='orange', label='Data Test (Aktual)')
        for nama, v in hasil_eval.items():
            ax_eval.plot(
                v['forecast'].index, v['forecast'].values,
                linestyle='--', marker='s',
                color=warna_metode[nama],
                label=f'Forecast {nama}'
            )
        ax_eval.axvline(x=test.index[0], color='gray', linestyle=':', alpha=0.7, label='Awal Periode Test')
        ax_eval.set_title(f'Evaluasi Gabungan Semua Metode — Produk {produk}')
        ax_eval.legend(loc='upper left')
        ax_eval.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_eval)

        # ==========================================
        # TABEL FORECAST KE DEPAN — SEMUA METODE
        # ==========================================

        st.subheader("🔮 Forecast ke Depan — Semua Metode")
        st.info(
            f"Semua model di-retrain menggunakan seluruh {len(data_produk)} bulan data, "
            "lalu meramalkan bulan-bulan berikutnya."
        )

        future_index = list(hasil_future.values())[0].index
        future_combined = pd.DataFrame({'Periode': future_index.strftime('%b-%Y')})
        for nama, fc in hasil_future.items():
            future_combined[nama] = np.round(fc.values, 2)
        future_combined.index = range(1, len(future_combined) + 1)

        st.dataframe(future_combined, use_container_width=True)

        csv_future = future_combined.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Forecast ke Depan (Semua Metode)",
            data=csv_future,
            file_name=f'forecast_ke_depan_{produk}_semua_metode.csv',
            mime='text/csv'
        )

        # ==========================================
        # GRAFIK FORECAST KE DEPAN GABUNGAN
        # Data aktual 36 bln + forecast ke depan tiap metode
        # ==========================================

        fig_future_all, ax_future_all = plt.subplots(figsize=(14, 6))
        ax_future_all.plot(
            data_produk.index, data_produk.values,
            marker='o', linewidth=2, color='steelblue',
            label=f'Data Aktual ({len(data_produk)} bulan)'
        )
        for nama, fc in hasil_future.items():
            ax_future_all.plot(
                fc.index, fc.values,
                linestyle='--', marker='s',
                color=warna_metode[nama],
                label=f'Forecast — {nama}'
            )
        ax_future_all.axvline(
            x=list(hasil_future.values())[0].index[0],
            color='red', linestyle='--', alpha=0.7, label='Awal Periode Forecast'
        )
        ax_future_all.set_title(f'Forecast ke Depan Gabungan Semua Metode — Produk {produk}')
        ax_future_all.legend(loc='upper left')
        ax_future_all.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_future_all)

        # ==========================================
        # GRAFIK FORECAST KE DEPAN — METODE TERBAIK
        # ==========================================

        st.subheader(f"🏆 Forecast ke Depan — Metode Terbaik: {metode_terbaik['Metode']}")

        fc_best = hasil_future[metode_terbaik['Metode']]

        best_df = pd.DataFrame({
            'Periode': fc_best.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(fc_best.values, 2)
        })
        best_df.index = range(1, len(best_df) + 1)
        st.dataframe(best_df, use_container_width=True)

        csv_best = best_df.to_csv().encode('utf-8')
        st.download_button(
            label=f"⬇️ Download Forecast Terbaik ({metode_terbaik['Metode']})",
            data=csv_best,
            file_name=f"forecast_terbaik_{produk}_{metode_terbaik['Metode']}.csv",
            mime='text/csv'
        )

        fig_best, ax_best = plt.subplots(figsize=(12, 5))
        ax_best.plot(
            data_produk.index, data_produk.values,
            marker='o', linewidth=2, color='steelblue',
            label=f'Data Aktual ({len(data_produk)} bulan)'
        )
        ax_best.plot(
            fc_best.index, fc_best.values,
            marker='o', linestyle='--',
            color=warna_metode[metode_terbaik['Metode']],
            linewidth=2,
            label=f"Forecast — {metode_terbaik['Metode']} (Terbaik)"
        )
        ax_best.axvline(x=fc_best.index[0], color='red', linestyle='--', alpha=0.7, label='Awal Periode Forecast')
        ax_best.set_title(f"Forecast ke Depan — {metode_terbaik['Metode']} (Terbaik) — Produk {produk}")
        ax_best.legend()
        ax_best.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_best)
