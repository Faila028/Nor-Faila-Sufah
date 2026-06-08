# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
# STYLE GRAFIK GLOBAL
# ==========================================

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'legend.framealpha':0.8,
    'figure.dpi':       120,
})

# Palet warna konsisten
WARNA = {
    'aktual':    '#2c7bb6',
    'train':     '#2c7bb6',
    'test':      '#f4a261',
    'HW Additive':       '#2ca02c',
    'HW Multiplicative': '#d62728',
    'ETS':               '#9467bd',
    'ARIMA':             '#8c564b',
    'vline':     '#aaaaaa',
}

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Terapkan style seragam ke axes."""
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc')
    ax.grid(axis='x', linestyle='',   alpha=0)
    ax.tick_params(axis='x', rotation=30)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# ==========================================
# JUDUL
# ==========================================

st.title("📦 Clustering dan Forecasting Permintaan Barang")

st.markdown("""
### Sistem Analisis Barang Keluar — PT. Menara Kudus Indonesia

Aplikasi ini dirancang untuk membantu tim gudang dan manajemen dalam menganalisis
pola permintaan barang secara data-driven, sehingga pengadaan barang dapat
dilakukan lebih efisien dan tepat sasaran.

**Fitur utama:**
- 🔵 **Clustering K-Means** — Mengelompokkan produk menjadi Fast, Medium, dan Slow Moving
- 📈 **Forecasting** — Meramalkan permintaan menggunakan Holt-Winters, ETS, dan ARIMA
- 📊 **Perbandingan Metode** — Memilih metode terbaik berdasarkan nilai MAE & RMSE

**Cara penggunaan:**
1. Upload file Excel data barang keluar
2. Tentukan jumlah cluster dan pilih produk
3. Pilih metode forecasting dan jumlah bulan ramalan
""")

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Awal")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", len(df))
    with col2:
        st.metric("Jumlah Produk", df['id_produk'].nunique())
    with col3:
        st.metric("Total Barang Keluar", int(df['keluar'].sum()))

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])
    df['Bulan']     = df['tgl_input'].dt.strftime('%b-%y')

    pivot_table = df.pivot_table(
        index='id_produk', columns='Bulan',
        values='keluar', aggfunc='sum', fill_value=0
    )

    urutan_bulan = [
        'Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23',
        'Jul-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24','May-24','Jun-24',
        'Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24',
        'Jan-25','Feb-25','Mar-25','Apr-25','May-25','Jun-25',
        'Jul-25','Aug-25','Sep-25','Oct-25','Nov-25','Dec-25',
    ]

    pivot_table = pivot_table.reindex(columns=urutan_bulan, fill_value=0)

    st.subheader("📊 Pivot Table Barang Keluar")
    st.dataframe(pivot_table)

    csv_data = pivot_table.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Pivot Table", csv_data,
                       'data_barang_keluar.csv', 'text/csv')

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.header("📌 Clustering Produk")

    pivot_table['Total'] = pivot_table.sum(axis=1)
    filtered_data = pivot_table[pivot_table['Total'] > 1].copy()

    scaler    = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data.drop(columns=['Total']))

    # ELBOW METHOD
    inertia = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertia.append(km.inertia_)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(list(K), inertia, marker='o', color=WARNA['aktual'],
             linewidth=2, markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(list(K), inertia, alpha=0.08, color=WARNA['aktual'])
    style_ax(ax1, title='Metode Elbow — Penentuan Jumlah Cluster Optimal',
             xlabel='Jumlah Cluster', ylabel='Inertia')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    jumlah_cluster = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)

    kmeans  = KMeans(n_clusters=jumlah_cluster, random_state=42)
    cluster = kmeans.fit_predict(scaled_data)
    filtered_data['Cluster'] = cluster

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
        'Kategori':      cluster_count.index,
        'Jumlah Produk': cluster_count.values
    })
    tabel_cluster.index = range(1, len(tabel_cluster) + 1)

    st.subheader("📊 Jumlah Produk per Cluster")
    st.dataframe(tabel_cluster, use_container_width=True)

    # Grafik cluster — horizontal bar lebih mudah dibaca
    warna_cluster = ['#2c7bb6', '#f4a261', '#d9534f']
    fig_cl, ax_cl = plt.subplots(figsize=(7, 3.5))
    bars = ax_cl.barh(
        tabel_cluster['Kategori'][::-1],
        tabel_cluster['Jumlah Produk'][::-1],
        color=warna_cluster[::-1],
        height=0.5,
        edgecolor='white'
    )
    ax_cl.bar_label(bars, fmt='%d', padding=5, fontsize=11, fontweight='bold')
    style_ax(ax_cl, title='Distribusi Produk per Cluster', xlabel='Jumlah Produk')
    ax_cl.grid(axis='x', linestyle='--', alpha=0.4, color='#cccccc')
    ax_cl.grid(axis='y', linestyle='', alpha=0)
    ax_cl.tick_params(axis='x', rotation=0)
    ax_cl.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax_cl.set_xlim(0, tabel_cluster['Jumlah Produk'].max() * 1.2)
    plt.tight_layout()
    st.pyplot(fig_cl)
    plt.close(fig_cl)

    pilih_cluster = st.selectbox(
        "Pilih Cluster",
        ['Fast Moving', 'Medium Moving', 'Slow Moving']
    )

    produk_cluster = filtered_data[
        filtered_data['Kategori'] == pilih_cluster
    ].index.tolist()

    st.subheader(f"📦 Produk dalam {pilih_cluster}")
    df_pc = pd.DataFrame({'Produk': produk_cluster})
    df_pc.index = range(1, len(df_pc) + 1)
    st.dataframe(df_pc, use_container_width=True)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    produk = st.selectbox("Pilih Produk", produk_cluster)

    data_produk = filtered_data.loc[produk].copy()
    data_produk = data_produk.drop(['Cluster', 'Kategori', 'Total'])
    data_produk = pd.to_numeric(data_produk)

    if (data_produk > 0).sum() < 6:
        st.warning("Produk ini hanya aktif kurang dari 6 bulan sehingga hasil forecasting kurang reliabel.")

    data_produk.index = pd.date_range(start='2023-01-01', periods=len(data_produk), freq='ME')

    metode = st.selectbox(
        "Pilih Metode Forecasting",
        ["Holt-Winters Additive", "Holt-Winters Multiplicative",
         "ETS", "ARIMA", "Perbandingan Semua Metode"]
    )

    jumlah_forecast = st.slider("Jumlah Forecast Bulan", 1, 12, 6)

    n     = len(data_produk)
    train = data_produk.iloc[:n - jumlah_forecast]
    test  = data_produk.iloc[n - jumlah_forecast:]

    if len(train) < 12:
        st.error(f"Data train hanya {len(train)} bulan. Kurangi jumlah forecast agar minimal 12 bulan.")
        st.stop()

    st.info(f"📌 Train: {len(train)} bulan  |  Test: {len(test)} bulan  |  MAE & RMSE dihitung dari forecast vs data test.")

    # Grafik data aktual
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(data_produk.index, data_produk.values,
             color=WARNA['aktual'], linewidth=2, marker='o',
             markersize=5, label='Data Aktual')
    ax2.axvspan(test.index[0], test.index[-1], alpha=0.08,
                color='orange', label=f'Periode Test ({len(test)} bln)')
    ax2.axvline(x=test.index[0], color=WARNA['vline'],
                linestyle='--', linewidth=1.2)
    style_ax(ax2, title=f'Data Aktual — Produk {produk}', ylabel='Jumlah Keluar')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ==========================================
    # HELPER: RETRAIN FULL DATA
    # ==========================================

    def retrain_full(nama_metode, full_data, n_fc):
        if nama_metode == 'HW Additive':
            return ExponentialSmoothing(
                full_data, trend='add', seasonal='add', seasonal_periods=12
            ).fit().forecast(n_fc).clip(lower=0)
        elif nama_metode == 'HW Multiplicative':
            fd = full_data.copy(); fd[fd <= 0] = 1
            return ExponentialSmoothing(
                fd, trend='add', seasonal='mul', seasonal_periods=12
            ).fit().forecast(n_fc).clip(lower=0)
        elif nama_metode == 'ETS':
            return ETSModel(
                full_data, error="add", trend="add", seasonal="add", seasonal_periods=12
            ).fit(disp=False).forecast(n_fc).clip(lower=0)
        elif nama_metode == 'ARIMA':
            return ARIMA(full_data, order=(1, 1, 1)).fit().forecast(steps=n_fc)

    # ==========================================
    # FUNGSI TAMPILKAN HASIL (1 METODE)
    # ==========================================

    def tampilkan_hasil(nama_metode, fc_eval, test_actual):

        mae  = mean_absolute_error(test_actual.values, fc_eval.values)
        rmse = np.sqrt(mean_squared_error(test_actual.values, fc_eval.values))

        c1, c2 = st.columns(2)
        with c1: st.metric("MAE",  f"{mae:.2f}")
        with c2: st.metric("RMSE", f"{rmse:.2f}")

        eval_df = pd.DataFrame({
            'Periode':              fc_eval.index.strftime('%b-%Y'),
            'Hasil Forecast':       np.round(fc_eval.values, 2),
            'Data Aktual (Test)':   np.round(test_actual.values, 2),
        })
        eval_df.index = range(1, len(eval_df) + 1)
        st.subheader("📋 Hasil Forecast vs Data Aktual (Evaluasi)")
        st.dataframe(eval_df, use_container_width=True)

        # Grafik evaluasi
        fig_e, ax_e = plt.subplots(figsize=(12, 4))
        ax_e.plot(train.index, train.values,
                  color=WARNA['train'], linewidth=2, marker='o',
                  markersize=4, label='Data Train')
        ax_e.plot(test_actual.index, test_actual.values,
                  color=WARNA['test'], linewidth=2, marker='o',
                  markersize=6, label='Data Test (Aktual)')
        ax_e.plot(fc_eval.index, fc_eval.values,
                  color=WARNA.get(nama_metode, 'green'),
                  linewidth=2, marker='s', markersize=6,
                  linestyle='--', label=f'Forecast {nama_metode}')
        ax_e.axvline(x=test_actual.index[0], color=WARNA['vline'],
                     linestyle='--', linewidth=1.2, label='Awal Test')
        style_ax(ax_e, title=f'Evaluasi — {nama_metode} — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_e.legend()
        plt.tight_layout()
        st.pyplot(fig_e)
        plt.close(fig_e)

        # Forecast ke depan
        st.subheader("🔮 Forecast ke Depan")
        st.info(f"Model di-retrain menggunakan seluruh {len(data_produk)} bulan data lalu meramalkan bulan berikutnya.")

        fc_future = retrain_full(nama_metode, data_produk, jumlah_forecast)

        fut_df = pd.DataFrame({
            'Periode':        fc_future.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(fc_future.values, 2),
        })
        fut_df.index = range(1, len(fut_df) + 1)
        st.dataframe(fut_df, use_container_width=True)

        csv_f = fut_df.to_csv().encode('utf-8')
        st.download_button("⬇️ Download Forecast ke Depan", csv_f,
                           f'forecast_{produk}_{nama_metode}.csv', 'text/csv')

        fig_f, ax_f = plt.subplots(figsize=(12, 4))
        ax_f.plot(data_produk.index, data_produk.values,
                  color=WARNA['aktual'], linewidth=2, marker='o',
                  markersize=4, label=f'Data Aktual ({len(data_produk)} bln)')
        ax_f.plot(fc_future.index, fc_future.values,
                  color=WARNA.get(nama_metode, 'green'),
                  linewidth=2.5, marker='s', markersize=7,
                  linestyle='--', label=f'Forecast ke Depan ({nama_metode})')
        ax_f.axvline(x=fc_future.index[0], color='red',
                     linestyle='--', linewidth=1.2, alpha=0.7, label='Awal Forecast')
        ax_f.axvspan(fc_future.index[0], fc_future.index[-1],
                     alpha=0.06, color='green')
        style_ax(ax_f, title=f'Forecast ke Depan — {nama_metode} — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_f.legend()
        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

        return mae, rmse

    # ==========================================
    # DISPATCH METODE TUNGGAL
    # ==========================================

    if metode == "Holt-Winters Additive":
        fc = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        tampilkan_hasil("HW Additive", fc, test)

    elif metode == "Holt-Winters Multiplicative":
        tr_nz = train.copy(); tr_nz[tr_nz <= 0] = 1
        fc = ExponentialSmoothing(tr_nz, trend='add', seasonal='mul', seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        tampilkan_hasil("HW Multiplicative", fc, test)

    elif metode == "ETS":
        fc = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)
        tampilkan_hasil("ETS", fc, test)

    elif metode == "ARIMA":
        fc = ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=jumlah_forecast)
        tampilkan_hasil("ARIMA", fc, test)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        hasil_eval   = {}
        hasil_future = {}

        # HW Additive
        fc_hwa = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Additive'] = {'forecast': fc_hwa,
            'mae':  mean_absolute_error(test.values, fc_hwa.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwa.values))}
        hasil_future['HW Additive'] = retrain_full('HW Additive', data_produk, jumlah_forecast)

        # HW Multiplicative
        tr_nz = train.copy(); tr_nz[tr_nz <= 0] = 1
        fc_hwm = ExponentialSmoothing(tr_nz, trend='add', seasonal='mul', seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Multiplicative'] = {'forecast': fc_hwm,
            'mae':  mean_absolute_error(test.values, fc_hwm.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwm.values))}
        hasil_future['HW Multiplicative'] = retrain_full('HW Multiplicative', data_produk, jumlah_forecast)

        # ETS
        fc_ets = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['ETS'] = {'forecast': fc_ets,
            'mae':  mean_absolute_error(test.values, fc_ets.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_ets.values))}
        hasil_future['ETS'] = retrain_full('ETS', data_produk, jumlah_forecast)

        # ARIMA
        fc_arima = ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=jumlah_forecast)
        hasil_eval['ARIMA'] = {'forecast': fc_arima,
            'mae':  mean_absolute_error(test.values, fc_arima.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_arima.values))}
        hasil_future['ARIMA'] = retrain_full('ARIMA', data_produk, jumlah_forecast)

        # Tabel perbandingan
        perbandingan = pd.DataFrame([
            {'Metode': m, 'MAE': round(v['mae'], 2), 'RMSE': round(v['rmse'], 2)}
            for m, v in hasil_eval.items()
        ])
        perbandingan.index = range(1, len(perbandingan) + 1)

        st.subheader("📊 Perbandingan MAE & RMSE Semua Metode")
        st.dataframe(perbandingan, use_container_width=True)

        # Grafik MAE & RMSE berdampingan
        fig_cmp, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(13, 4))

        metode_list  = perbandingan['Metode'].tolist()
        warna_list   = [WARNA[m] for m in metode_list]

        b1 = ax_mae.bar(metode_list, perbandingan['MAE'],
                        color=warna_list, edgecolor='white', width=0.5)
        ax_mae.bar_label(b1, fmt='%.2f', padding=4, fontsize=10, fontweight='bold')
        style_ax(ax_mae, title='Perbandingan MAE', ylabel='MAE')
        ax_mae.set_ylim(0, perbandingan['MAE'].max() * 1.25)

        b2 = ax_rmse.bar(metode_list, perbandingan['RMSE'],
                         color=warna_list, edgecolor='white', width=0.5)
        ax_rmse.bar_label(b2, fmt='%.2f', padding=4, fontsize=10, fontweight='bold')
        style_ax(ax_rmse, title='Perbandingan RMSE', ylabel='RMSE')
        ax_rmse.set_ylim(0, perbandingan['RMSE'].max() * 1.25)

        plt.suptitle('Evaluasi Akurasi Semua Metode', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        # Metode terbaik
        best_idx       = perbandingan['MAE'].idxmin()
        metode_terbaik = perbandingan.loc[best_idx]
        st.success(
            f"✅ Metode terbaik: **{metode_terbaik['Metode']}**  |  "
            f"MAE = **{metode_terbaik['MAE']:.2f}**  |  "
            f"RMSE = **{metode_terbaik['RMSE']:.2f}**"
        )

        # Grafik evaluasi gabungan
        st.subheader("📉 Grafik Evaluasi Gabungan (Train vs Test vs Forecast)")

        fig_ev, ax_ev = plt.subplots(figsize=(13, 5))
        ax_ev.plot(train.index, train.values, color=WARNA['train'],
                   linewidth=2, marker='o', markersize=4, label='Data Train')
        ax_ev.plot(test.index, test.values, color=WARNA['test'],
                   linewidth=2.5, marker='o', markersize=7, label='Data Test (Aktual)')
        for nm, v in hasil_eval.items():
            ax_ev.plot(v['forecast'].index, v['forecast'].values,
                       color=WARNA[nm], linewidth=1.8, marker='s',
                       markersize=5, linestyle='--', label=f'Forecast {nm}')
        ax_ev.axvline(x=test.index[0], color=WARNA['vline'],
                      linestyle='--', linewidth=1.2, label='Awal Test')
        ax_ev.axvspan(test.index[0], test.index[-1], alpha=0.06, color='orange')
        style_ax(ax_ev, title=f'Evaluasi Gabungan — Produk {produk}', ylabel='Jumlah Keluar')
        ax_ev.legend(loc='upper left', ncol=2)
        plt.tight_layout()
        st.pyplot(fig_ev)
        plt.close(fig_ev)

        # Tabel forecast ke depan semua metode
        st.subheader("🔮 Forecast ke Depan — Semua Metode")
        st.info(f"Semua model di-retrain menggunakan seluruh {len(data_produk)} bulan data.")

        fut_idx  = list(hasil_future.values())[0].index
        fut_comb = pd.DataFrame({'Periode': fut_idx.strftime('%b-%Y')})
        for nm, fc in hasil_future.items():
            fut_comb[nm] = np.round(fc.values, 2)
        fut_comb.index = range(1, len(fut_comb) + 1)
        st.dataframe(fut_comb, use_container_width=True)

        csv_fut = fut_comb.to_csv().encode('utf-8')
        st.download_button("⬇️ Download Forecast ke Depan (Semua Metode)", csv_fut,
                           f'forecast_ke_depan_{produk}_semua.csv', 'text/csv')

        # Grafik forecast ke depan gabungan
        fig_fa, ax_fa = plt.subplots(figsize=(13, 5))
        ax_fa.plot(data_produk.index, data_produk.values,
                   color=WARNA['aktual'], linewidth=2, marker='o',
                   markersize=4, label=f'Data Aktual ({len(data_produk)} bln)')
        for nm, fc in hasil_future.items():
            ax_fa.plot(fc.index, fc.values, color=WARNA[nm],
                       linewidth=1.8, marker='s', markersize=5,
                       linestyle='--', label=f'Forecast — {nm}')
        first_fc = list(hasil_future.values())[0]
        ax_fa.axvline(x=first_fc.index[0], color='red',
                      linestyle='--', linewidth=1.2, alpha=0.7, label='Awal Forecast')
        ax_fa.axvspan(first_fc.index[0], first_fc.index[-1], alpha=0.06, color='green')
        style_ax(ax_fa, title=f'Forecast ke Depan Gabungan — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_fa.legend(loc='upper left', ncol=2)
        plt.tight_layout()
        st.pyplot(fig_fa)
        plt.close(fig_fa)

        # Grafik metode terbaik
        st.subheader(f"🏆 Forecast ke Depan — Metode Terbaik: {metode_terbaik['Metode']}")

        fc_best = hasil_future[metode_terbaik['Metode']]
        best_df = pd.DataFrame({
            'Periode':        fc_best.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(fc_best.values, 2),
        })
        best_df.index = range(1, len(best_df) + 1)
        st.dataframe(best_df, use_container_width=True)

        csv_best = best_df.to_csv().encode('utf-8')
        st.download_button(
            f"⬇️ Download Forecast Terbaik ({metode_terbaik['Metode']})",
            csv_best, f"forecast_terbaik_{produk}.csv", 'text/csv'
        )

        fig_bst, ax_bst = plt.subplots(figsize=(12, 4))
        ax_bst.plot(data_produk.index, data_produk.values,
                    color=WARNA['aktual'], linewidth=2, marker='o',
                    markersize=4, label=f'Data Aktual ({len(data_produk)} bln)')
        ax_bst.plot(fc_best.index, fc_best.values,
                    color=WARNA[metode_terbaik['Metode']],
                    linewidth=2.5, marker='s', markersize=8,
                    linestyle='--', label=f"Forecast — {metode_terbaik['Metode']} (Terbaik)")
        ax_bst.axvline(x=fc_best.index[0], color='red',
                       linestyle='--', linewidth=1.2, alpha=0.7, label='Awal Forecast')
        ax_bst.axvspan(fc_best.index[0], fc_best.index[-1], alpha=0.08,
                       color=WARNA[metode_terbaik['Metode']])
        style_ax(ax_bst,
                 title=f"Forecast Terbaik — {metode_terbaik['Metode']} — Produk {produk}",
                 ylabel='Jumlah Keluar')
        ax_bst.legend()
        plt.tight_layout()
        st.pyplot(fig_bst)
        plt.close(fig_bst)
