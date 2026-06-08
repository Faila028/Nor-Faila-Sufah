# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# ==========================================
# CONFIG & STYLE GLOBAL
# ==========================================

st.set_page_config(page_title="Forecasting Barang", page_icon="📦", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7fa; }
h1 { color: #1f4e79; text-align: center; font-weight: bold; }
h2, h3 { color: #1f4e79; }
.stButton>button {
    background-color: #1f77b4; color: white;
    border-radius: 10px; border: none;
    padding: 10px 24px; font-weight: bold;
}
.stDownloadButton>button {
    background-color: #28a745; color: white;
    border-radius: 10px; border: none;
    padding: 10px 24px; font-weight: bold;
}
[data-testid="stMetricValue"] { color: #1f77b4; font-size: 28px; }
div[data-testid="stTabs"] button { font-weight: 600; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'legend.fontsize': 10,
    'legend.framealpha': 0.8, 'figure.dpi': 120,
})

WARNA = {
    'aktual': '#2c7bb6', 'train': '#2c7bb6', 'test': '#f4a261',
    'HW Additive': '#2ca02c', 'HW Multiplicative': '#d62728',
    'ETS': '#9467bd', 'ARIMA': '#8c564b', 'vline': '#aaaaaa',
}

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, pad=12)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc')
    ax.grid(axis='x', linestyle='', alpha=0)
    ax.tick_params(axis='x', rotation=30)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# ==========================================
# JUDUL
# ==========================================

st.title("📦 Clustering dan Forecasting Permintaan Barang")
st.markdown("""
### Sistem Analisis Barang Keluar — PT. Menara Kudus Indonesia
Aplikasi berbasis web untuk mengelompokkan produk berdasarkan tingkat pergerakan
(*Fast / Medium / Slow Moving*) dan meramalkan permintaan barang menggunakan
metode Holt-Winters, ETS, dan ARIMA — guna mendukung pengendalian persediaan.
""")

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader("📂 Upload File Excel Data Barang Keluar", type=["xlsx"])

if uploaded_file is None:
    st.info("👆 Silakan upload file Excel terlebih dahulu untuk memulai analisis.")
    st.stop()

# ==========================================
# BACA DATA
# ==========================================

df = pd.read_excel(uploaded_file)
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

# ==========================================
# TABS UTAMA
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📊  Data & Pivot Table",
    "📌  Clustering Produk",
    "📈  Forecasting",
])

# ==========================================
# TAB 1 — DATA & PIVOT TABLE
# ==========================================

with tab1:
    st.subheader("📄 Data Awal")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Jumlah Data",        len(df))
    with c2: st.metric("Jumlah Produk",      df['id_produk'].nunique())
    with c3: st.metric("Total Barang Keluar", int(df['keluar'].sum()))

    st.subheader("📊 Pivot Table Barang Keluar per Bulan")
    st.caption("Setiap baris = 1 produk, setiap kolom = total keluar per bulan.")
    st.dataframe(pivot_table, use_container_width=True)

    csv_pivot = pivot_table.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Pivot Table", csv_pivot,
                       'pivot_barang_keluar.csv', 'text/csv')

# ==========================================
# PERSIAPAN CLUSTERING (dipakai Tab 2 & 3)
# ==========================================

pivot_table['Total'] = pivot_table.sum(axis=1)
filtered_data        = pivot_table[pivot_table['Total'] > 1].copy()
scaler               = StandardScaler()
scaled_data          = scaler.fit_transform(filtered_data.drop(columns=['Total']))

# ==========================================
# TAB 2 — CLUSTERING
# ==========================================

with tab2:

    st.subheader("📌 Clustering Produk dengan K-Means")

    # Elbow method
    inertia = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertia.append(km.inertia_)

    fig_el, ax_el = plt.subplots(figsize=(7, 4))
    ax_el.plot(range(1, 10), inertia, marker='o', color=WARNA['aktual'],
               linewidth=2, markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax_el.fill_between(range(1, 10), inertia, alpha=0.08, color=WARNA['aktual'])
    style_ax(ax_el, title='Metode Elbow — Penentuan Jumlah Cluster Optimal',
             xlabel='Jumlah Cluster', ylabel='Inertia')
    ax_el.grid(axis='x', linestyle='--', alpha=0.3)
    ax_el.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig_el)
    plt.close(fig_el)

    st.caption("💡 Pilih jumlah cluster di titik 'siku' — di mana penurunan inertia mulai melambat.")

    jumlah_cluster = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)

    kmeans  = KMeans(n_clusters=jumlah_cluster, random_state=42)
    cluster = kmeans.fit_predict(scaled_data)
    filtered_data = filtered_data.copy()
    filtered_data['Cluster'] = cluster

    cluster_avg = filtered_data.groupby('Cluster')['Total'].mean().sort_values(ascending=False)
    mapping_cluster = {}
    labels_tersedia = ['Fast Moving', 'Medium Moving', 'Slow Moving']
    for i, idx in enumerate(cluster_avg.index[:3]):
        mapping_cluster[idx] = labels_tersedia[i]

    filtered_data['Kategori'] = filtered_data['Cluster'].map(mapping_cluster)

    cluster_count = filtered_data['Kategori'].value_counts().reindex(
        ['Fast Moving', 'Medium Moving', 'Slow Moving']
    )
    tabel_cluster = pd.DataFrame({
        'Kategori':      cluster_count.index,
        'Jumlah Produk': cluster_count.values,
    })
    tabel_cluster.index = range(1, len(tabel_cluster) + 1)

    col_tbl, col_chart = st.columns([1, 1.5])

    with col_tbl:
        st.subheader("Jumlah Produk per Cluster")
        st.dataframe(tabel_cluster, use_container_width=True)

    with col_chart:
        warna_cl = ['#2c7bb6', '#f4a261', '#d9534f']
        fig_cl, ax_cl = plt.subplots(figsize=(6, 3.2))
        bars = ax_cl.barh(
            tabel_cluster['Kategori'][::-1],
            tabel_cluster['Jumlah Produk'][::-1],
            color=warna_cl[::-1], height=0.5, edgecolor='white'
        )
        ax_cl.bar_label(bars, fmt='%d', padding=5, fontsize=11, fontweight='bold')
        style_ax(ax_cl, title='Distribusi Produk per Cluster', xlabel='Jumlah Produk')
        ax_cl.grid(axis='x', linestyle='--', alpha=0.4, color='#cccccc')
        ax_cl.grid(axis='y', linestyle='', alpha=0)
        ax_cl.tick_params(axis='x', rotation=0)
        ax_cl.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax_cl.set_xlim(0, tabel_cluster['Jumlah Produk'].max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig_cl)
        plt.close(fig_cl)

    # Daftar produk per cluster
    st.subheader("Daftar Produk per Cluster")
    pilih_lihat = st.selectbox("Tampilkan produk dalam cluster:",
                               ['Fast Moving', 'Medium Moving', 'Slow Moving'],
                               key='lihat_cluster')
    produk_lihat = filtered_data[filtered_data['Kategori'] == pilih_lihat].index.tolist()
    df_lihat = pd.DataFrame({'Produk': produk_lihat})
    df_lihat.index = range(1, len(df_lihat) + 1)
    st.dataframe(df_lihat, use_container_width=True)

    # Simpan ke session state agar Tab 3 bisa pakai
    st.session_state['filtered_data']  = filtered_data
    st.session_state['clustering_done'] = True

# ==========================================
# TAB 3 — FORECASTING
# ==========================================

with tab3:

    if not st.session_state.get('clustering_done'):
        st.warning("⚠️ Selesaikan dulu tab **Clustering Produk** sebelum melakukan forecasting.")
        st.stop()

    filtered_data = st.session_state['filtered_data']

    st.subheader("📈 Forecasting Permintaan Barang")

    # --- Pilihan ---
    col_a, col_b = st.columns(2)

    with col_a:
        pilih_cluster = st.selectbox(
            "Pilih Cluster", ['Fast Moving', 'Medium Moving', 'Slow Moving']
        )
        produk_cluster = filtered_data[
            filtered_data['Kategori'] == pilih_cluster
        ].index.tolist()
        produk = st.selectbox("Pilih Produk", produk_cluster)

    with col_b:
        metode = st.selectbox(
            "Pilih Metode Forecasting",
            ["Holt-Winters Additive", "Holt-Winters Multiplicative",
             "ETS", "ARIMA", "Perbandingan Semua Metode"]
        )
        jumlah_forecast = st.slider("Jumlah Bulan yang Diramalkan", 1, 12, 6)

    # Tombol jalankan
    st.markdown("---")
    run = st.button("🚀 Jalankan Forecast", use_container_width=True)

    if not run:
        st.info("👆 Atur pilihan di atas lalu klik **Jalankan Forecast**.")
        st.stop()

    # --- Siapkan data produk ---
    data_produk = filtered_data.loc[produk].copy()
    data_produk = data_produk.drop(['Cluster', 'Kategori', 'Total'])
    data_produk = pd.to_numeric(data_produk)
    data_produk.index = pd.date_range(start='2023-01-01',
                                       periods=len(data_produk), freq='ME')

    if (data_produk > 0).sum() < 6:
        st.warning("⚠️ Produk ini aktif kurang dari 6 bulan — hasil forecast kurang reliabel.")

    # Train-test split
    n     = len(data_produk)
    train = data_produk.iloc[:n - jumlah_forecast]
    test  = data_produk.iloc[n - jumlah_forecast:]

    if len(train) < 12:
        st.error(f"Data train hanya {len(train)} bulan. Kurangi jumlah forecast agar minimal 12 bulan.")
        st.stop()

    st.info(f"📌 Train: **{len(train)} bulan**  |  Test: **{len(test)} bulan**  |  "
            "MAE & RMSE dihitung dari hasil forecast vs data test.")

    # Grafik data aktual
    fig_akt, ax_akt = plt.subplots(figsize=(12, 4))
    ax_akt.plot(data_produk.index, data_produk.values,
                color=WARNA['aktual'], linewidth=2, marker='o', markersize=5, label='Data Aktual')
    ax_akt.axvspan(test.index[0], test.index[-1], alpha=0.08, color='orange',
                   label=f'Periode Test ({len(test)} bln)')
    ax_akt.axvline(x=test.index[0], color=WARNA['vline'], linestyle='--', linewidth=1.2)
    style_ax(ax_akt, title=f'Data Aktual Barang Keluar — Produk {produk}',
             ylabel='Jumlah Keluar')
    ax_akt.legend()
    plt.tight_layout()
    st.pyplot(fig_akt)
    plt.close(fig_akt)

    # ==========================================
    # HELPER
    # ==========================================

    def retrain_full(nama, full_data, n_fc):
        if nama == 'HW Additive':
            return ExponentialSmoothing(full_data, trend='add', seasonal='add',
                                        seasonal_periods=12).fit().forecast(n_fc).clip(lower=0)
        elif nama == 'HW Multiplicative':
            fd = full_data.copy(); fd[fd <= 0] = 1
            return ExponentialSmoothing(fd, trend='add', seasonal='mul',
                                        seasonal_periods=12).fit().forecast(n_fc).clip(lower=0)
        elif nama == 'ETS':
            return ETSModel(full_data, error="add", trend="add", seasonal="add",
                            seasonal_periods=12).fit(disp=False).forecast(n_fc).clip(lower=0)
        elif nama == 'ARIMA':
            return ARIMA(full_data, order=(1, 1, 1)).fit().forecast(steps=n_fc)

    def tampilkan_satu_metode(nama, fc_eval, test_actual):
        mae  = mean_absolute_error(test_actual.values, fc_eval.values)
        rmse = np.sqrt(mean_squared_error(test_actual.values, fc_eval.values))

        c1, c2 = st.columns(2)
        with c1: st.metric("MAE",  f"{mae:.2f}")
        with c2: st.metric("RMSE", f"{rmse:.2f}")

        # Tabel evaluasi
        eval_df = pd.DataFrame({
            'Periode':             fc_eval.index.strftime('%b-%Y'),
            'Forecast (Evaluasi)': np.round(fc_eval.values, 2),
            'Data Aktual (Test)':  np.round(test_actual.values, 2),
        })
        eval_df.index = range(1, len(eval_df) + 1)
        st.subheader("📋 Perbandingan Forecast vs Aktual (Periode Evaluasi)")
        st.dataframe(eval_df, use_container_width=True)

        # Grafik evaluasi
        fig_e, ax_e = plt.subplots(figsize=(12, 4))
        ax_e.plot(train.index, train.values, color=WARNA['train'],
                  linewidth=2, marker='o', markersize=4, label='Data Train')
        ax_e.plot(test_actual.index, test_actual.values, color=WARNA['test'],
                  linewidth=2.5, marker='o', markersize=7, label='Data Test (Aktual)')
        ax_e.plot(fc_eval.index, fc_eval.values,
                  color=WARNA.get(nama, 'green'), linewidth=2,
                  marker='s', markersize=6, linestyle='--', label=f'Forecast {nama}')
        ax_e.axvline(x=test_actual.index[0], color=WARNA['vline'],
                     linestyle='--', linewidth=1.2, label='Awal Test')
        ax_e.axvspan(test_actual.index[0], test_actual.index[-1], alpha=0.06, color='orange')
        style_ax(ax_e, title=f'Evaluasi Model {nama} — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_e.legend()
        plt.tight_layout()
        st.pyplot(fig_e)
        plt.close(fig_e)

        # Forecast ke depan
        st.subheader("🔮 Forecast ke Depan")
        st.caption(f"Model di-retrain menggunakan seluruh {len(data_produk)} bulan data, "
                   "lalu meramalkan bulan-bulan berikutnya.")

        fc_future = retrain_full(nama, data_produk, jumlah_forecast)
        fut_df = pd.DataFrame({
            'Periode':        fc_future.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(fc_future.values, 2),
        })
        fut_df.index = range(1, len(fut_df) + 1)
        st.dataframe(fut_df, use_container_width=True)

        csv_f = fut_df.to_csv().encode('utf-8')
        st.download_button("⬇️ Download Forecast ke Depan", csv_f,
                           f'forecast_{produk}_{nama}.csv', 'text/csv')

        fig_f, ax_f = plt.subplots(figsize=(12, 4))
        ax_f.plot(data_produk.index, data_produk.values, color=WARNA['aktual'],
                  linewidth=2, marker='o', markersize=4,
                  label=f'Data Aktual ({len(data_produk)} bln)')
        ax_f.plot(fc_future.index, fc_future.values,
                  color=WARNA.get(nama, 'green'), linewidth=2.5,
                  marker='s', markersize=7, linestyle='--',
                  label=f'Forecast ke Depan ({nama})')
        ax_f.axvline(x=fc_future.index[0], color='red', linestyle='--',
                     linewidth=1.2, alpha=0.7, label='Awal Forecast')
        ax_f.axvspan(fc_future.index[0], fc_future.index[-1], alpha=0.07, color='green')
        style_ax(ax_f, title=f'Forecast ke Depan — {nama} — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_f.legend()
        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

        return mae, rmse, fc_future

    # ==========================================
    # METODE TUNGGAL
    # ==========================================

    if metode != "Perbandingan Semua Metode":

        nama_map = {
            "Holt-Winters Additive":       "HW Additive",
            "Holt-Winters Multiplicative": "HW Multiplicative",
            "ETS":   "ETS",
            "ARIMA": "ARIMA",
        }
        nama = nama_map[metode]

        if nama == "HW Additive":
            fc = ExponentialSmoothing(train, trend='add', seasonal='add',
                                      seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        elif nama == "HW Multiplicative":
            tr_nz = train.copy(); tr_nz[tr_nz <= 0] = 1
            fc = ExponentialSmoothing(tr_nz, trend='add', seasonal='mul',
                                      seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        elif nama == "ETS":
            fc = ETSModel(train, error="add", trend="add", seasonal="add",
                          seasonal_periods=12).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)
        elif nama == "ARIMA":
            fc = ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=jumlah_forecast)

        mae, rmse, fc_future = tampilkan_satu_metode(nama, fc, test)

        # Ringkasan
        st.markdown("---")
        st.subheader("📝 Ringkasan Hasil")
        st.success(
            f"**Produk:** {produk}  |  "
            f"**Cluster:** {pilih_cluster}  |  "
            f"**Metode:** {nama}  |  "
            f"**MAE:** {mae:.2f}  |  "
            f"**RMSE:** {rmse:.2f}  |  "
            f"**Forecast bulan pertama:** {int(round(fc_future.values[0]))} unit"
        )

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        hasil_eval   = {}
        hasil_future = {}

        # HW Additive
        fc_hwa = ExponentialSmoothing(train, trend='add', seasonal='add',
                                      seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Additive'] = {
            'forecast': fc_hwa,
            'mae':  mean_absolute_error(test.values, fc_hwa.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwa.values)),
        }
        hasil_future['HW Additive'] = retrain_full('HW Additive', data_produk, jumlah_forecast)

        # HW Multiplicative
        tr_nz = train.copy(); tr_nz[tr_nz <= 0] = 1
        fc_hwm = ExponentialSmoothing(tr_nz, trend='add', seasonal='mul',
                                      seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Multiplicative'] = {
            'forecast': fc_hwm,
            'mae':  mean_absolute_error(test.values, fc_hwm.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwm.values)),
        }
        hasil_future['HW Multiplicative'] = retrain_full('HW Multiplicative', data_produk, jumlah_forecast)

        # ETS
        fc_ets = ETSModel(train, error="add", trend="add", seasonal="add",
                          seasonal_periods=12).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['ETS'] = {
            'forecast': fc_ets,
            'mae':  mean_absolute_error(test.values, fc_ets.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_ets.values)),
        }
        hasil_future['ETS'] = retrain_full('ETS', data_produk, jumlah_forecast)

        # ARIMA
        fc_arima = ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=jumlah_forecast)
        hasil_eval['ARIMA'] = {
            'forecast': fc_arima,
            'mae':  mean_absolute_error(test.values, fc_arima.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_arima.values)),
        }
        hasil_future['ARIMA'] = retrain_full('ARIMA', data_produk, jumlah_forecast)

        # Tabel perbandingan
        perbandingan = pd.DataFrame([
            {'Metode': m, 'MAE': round(v['mae'], 2), 'RMSE': round(v['rmse'], 2)}
            for m, v in hasil_eval.items()
        ])
        perbandingan.index = range(1, len(perbandingan) + 1)

        best_idx       = perbandingan['MAE'].idxmin()
        metode_terbaik = perbandingan.loc[best_idx]

        st.subheader("📊 Perbandingan Akurasi Semua Metode")

        # Highlight baris terbaik
        def highlight_best(row):
            return ['background-color: #d4edda; font-weight: bold'
                    if row['Metode'] == metode_terbaik['Metode'] else '' for _ in row]

        st.dataframe(
            perbandingan.style.apply(highlight_best, axis=1),
            use_container_width=True
        )

        # Grafik MAE & RMSE berdampingan
        fig_cmp, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(13, 4))
        metode_list = perbandingan['Metode'].tolist()
        warna_list  = [WARNA[m] for m in metode_list]

        b1 = ax_mae.bar(metode_list, perbandingan['MAE'],
                        color=warna_list, edgecolor='white', width=0.5)
        ax_mae.bar_label(b1, fmt='%.2f', padding=4, fontsize=10, fontweight='bold')
        style_ax(ax_mae, title='Perbandingan MAE', ylabel='MAE')
        ax_mae.set_ylim(0, perbandingan['MAE'].max() * 1.3)

        b2 = ax_rmse.bar(metode_list, perbandingan['RMSE'],
                         color=warna_list, edgecolor='white', width=0.5)
        ax_rmse.bar_label(b2, fmt='%.2f', padding=4, fontsize=10, fontweight='bold')
        style_ax(ax_rmse, title='Perbandingan RMSE', ylabel='RMSE')
        ax_rmse.set_ylim(0, perbandingan['RMSE'].max() * 1.3)

        plt.suptitle('Evaluasi Akurasi Semua Metode', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.success(
            f"✅ Metode terbaik: **{metode_terbaik['Metode']}**  |  "
            f"MAE = **{metode_terbaik['MAE']:.2f}**  |  "
            f"RMSE = **{metode_terbaik['RMSE']:.2f}**"
        )

        # Grafik evaluasi gabungan
        st.subheader("📉 Grafik Evaluasi Gabungan")
        st.caption("Perbandingan hasil forecast setiap metode terhadap data test yang sebenarnya.")

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
        style_ax(ax_ev, title=f'Evaluasi Gabungan Semua Metode — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_ev.legend(loc='upper left', ncol=2)
        plt.tight_layout()
        st.pyplot(fig_ev)
        plt.close(fig_ev)

        # Forecast ke depan
        st.subheader("🔮 Forecast ke Depan — Semua Metode")
        st.caption(f"Semua model di-retrain menggunakan seluruh {len(data_produk)} bulan data.")

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
        ax_fa.plot(data_produk.index, data_produk.values, color=WARNA['aktual'],
                   linewidth=2, marker='o', markersize=4,
                   label=f'Data Aktual ({len(data_produk)} bln)')
        for nm, fc in hasil_future.items():
            ax_fa.plot(fc.index, fc.values, color=WARNA[nm], linewidth=1.8,
                       marker='s', markersize=5, linestyle='--', label=f'Forecast — {nm}')
        first_fc = list(hasil_future.values())[0]
        ax_fa.axvline(x=first_fc.index[0], color='red', linestyle='--',
                      linewidth=1.2, alpha=0.7, label='Awal Forecast')
        ax_fa.axvspan(first_fc.index[0], first_fc.index[-1], alpha=0.06, color='green')
        style_ax(ax_fa, title=f'Forecast ke Depan Gabungan Semua Metode — Produk {produk}',
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
        ax_bst.plot(data_produk.index, data_produk.values, color=WARNA['aktual'],
                    linewidth=2, marker='o', markersize=4,
                    label=f'Data Aktual ({len(data_produk)} bln)')
        ax_bst.plot(fc_best.index, fc_best.values,
                    color=WARNA[metode_terbaik['Metode']], linewidth=2.5,
                    marker='s', markersize=8, linestyle='--',
                    label=f"Forecast — {metode_terbaik['Metode']} (Terbaik)")
        ax_bst.axvline(x=fc_best.index[0], color='red', linestyle='--',
                       linewidth=1.2, alpha=0.7, label='Awal Forecast')
        ax_bst.axvspan(fc_best.index[0], fc_best.index[-1], alpha=0.08,
                       color=WARNA[metode_terbaik['Metode']])
        style_ax(ax_bst,
                 title=f"Forecast Terbaik — {metode_terbaik['Metode']} — Produk {produk}",
                 ylabel='Jumlah Keluar')
        ax_bst.legend()
        plt.tight_layout()
        st.pyplot(fig_bst)
        plt.close(fig_bst)

        # ==========================================
        # RINGKASAN AKHIR
        # ==========================================

        st.markdown("---")
        st.subheader("📝 Ringkasan Hasil Analisis")

        fc_best_val = int(round(fc_best.values[0]))

        st.success(
            f"**Produk:** {produk}  |  "
            f"**Cluster:** {pilih_cluster}  |  "
            f"**Metode Terbaik:** {metode_terbaik['Metode']}  |  "
            f"**MAE:** {metode_terbaik['MAE']:.2f}  |  "
            f"**RMSE:** {metode_terbaik['RMSE']:.2f}  |  "
            f"**Forecast bulan pertama:** {fc_best_val} unit"
        )

        st.caption(
            "💡 Gunakan hasil forecast metode terbaik sebagai acuan perencanaan pengadaan barang "
            "pada periode berikutnya."
        )
