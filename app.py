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

df_raw = pd.read_excel(uploaded_file)
df_raw['tgl_input'] = pd.to_datetime(df_raw['tgl_input'])
df_raw['Bulan']     = df_raw['tgl_input'].dt.strftime('%b-%y')

# ==========================================
# DETEKSI MISSING VALUES (SEBELUM CLEANING)
# ==========================================

missing_id_produk = df_raw['id_produk'].isna().sum()
missing_keluar    = df_raw['keluar'].isna().sum()
total_missing     = missing_id_produk + missing_keluar

# Produk yang tidak pernah terjual sama sekali (keluar = 0 atau NaN semua bulan)
# Hitung per produk: total keluar (NaN dianggap 0)
total_per_produk = df_raw.groupby('id_produk')['keluar'].sum(min_count=1).fillna(0)
produk_tidak_terjual = total_per_produk[total_per_produk == 0].index.tolist()

# ==========================================
# CLEANING DATA
# ==========================================

df = df_raw.dropna(subset=['id_produk', 'keluar']).copy()

# ==========================================
# PIVOT TABLE
# ==========================================

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
    st.dataframe(df_raw.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Jumlah Data",         f"{len(df_raw):,}")
    with c2: st.metric("Jumlah Produk",       df_raw['id_produk'].nunique())
    with c3: st.metric("Total Barang Keluar", f"{int(df['keluar'].sum()):,}")

    # ==========================================
    # SECTION: KUALITAS DATA & MISSING VALUES
    # ==========================================

    st.markdown("---")
    st.subheader("🔍 Kualitas Data & Missing Values")

    col_mv1, col_mv2, col_mv3 = st.columns(3)
    with col_mv1:
        st.metric(
            label="Baris ID Produk Kosong",
            value=f"{missing_id_produk:,}",
            delta="diabaikan" if missing_id_produk > 0 else "✅ lengkap",
            delta_color="off" if missing_id_produk > 0 else "normal"
        )
    with col_mv2:
        st.metric(
            label="Baris Nilai Keluar Kosong",
            value=f"{missing_keluar:,}",
            delta="diabaikan" if missing_keluar > 0 else "✅ lengkap",
            delta_color="off" if missing_keluar > 0 else "normal"
        )
    with col_mv3:
        st.metric(
            label="Total Baris Tidak Valid",
            value=f"{total_missing:,}",
            delta=f"tersisa {len(df):,} baris valid",
            delta_color="off"
        )

    if missing_id_produk > 0 or missing_keluar > 0:
        st.warning(
            f"⚠️ Ditemukan **{total_missing:,} baris tidak valid** "
            f"({missing_id_produk:,} baris ID produk kosong & "
            f"{missing_keluar:,} baris nilai keluar kosong). "
        )
    else:
        st.success("✅ Tidak ada missing values. Data siap dianalisis.")

    # ==========================================
    # PRODUK TIDAK PERNAH TERJUAL
    # ==========================================

    st.markdown("---")
    st.subheader("📭 Produk Tidak Pernah Terjual")

    if len(produk_tidak_terjual) > 0:
        df_tidak_terjual = pd.DataFrame({
            'No': range(1, len(produk_tidak_terjual) + 1),
            'ID Produk': produk_tidak_terjual
        }).set_index('No')

        st.warning(
            f"⚠️ Terdapat **{len(produk_tidak_terjual)} produk** dengan total barang keluar = 0 "
            "sepanjang periode Januari 2023 – Desember 2025."
        )
        st.info(
            "ℹ️ Produk-produk ini tidak diikutsertakan dalam proses analisis "
            "clustering maupun forecasting karena tidak memiliki data historis yang "
            "dapat digunakan untuk membentuk pola pergerakan barang."
        )
        st.dataframe(df_tidak_terjual, use_container_width=True)

        csv_tidak_terjual = df_tidak_terjual.to_csv().encode('utf-8')
        st.download_button(
            "⬇️ Download Daftar Produk Tidak Terjual",
            csv_tidak_terjual, 'produk_tidak_terjual.csv', 'text/csv'
        )
    else:
        st.success("✅ Semua produk memiliki setidaknya satu transaksi barang keluar.")

    # ==========================================
    # PIVOT TABLE
    # ==========================================

    st.markdown("---")
    st.subheader("📊 Pivot Table Barang Keluar per Bulan")
    st.caption("Setiap baris = 1 produk, setiap kolom = total keluar per bulan.")
    st.dataframe(pivot_table, use_container_width=True)

    csv_pivot = pivot_table.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Pivot Table", csv_pivot,
                       'pivot_barang_keluar.csv', 'text/csv')

    st.caption(f"📦 Total produk dalam pivot table: **{len(pivot_table):,} produk** "
               f"dari **{len(urutan_bulan)} bulan** periode Januari 2023 – Desember 2025.")

# ==========================================
# PERSIAPAN CLUSTERING
# ==========================================

pivot_table['Total'] = pivot_table.sum(axis=1)
filtered_data        = pivot_table.copy()
scaler               = StandardScaler()
scaled_data          = scaler.fit_transform(filtered_data.drop(columns=['Total']))

# ==========================================
# TAB 2 — CLUSTERING
# ==========================================

with tab2:

    st.subheader("📌 Clustering Produk dengan K-Means")
    st.markdown("---")

    inertia = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertia.append(km.inertia_)

    fig_el, ax_el = plt.subplots(figsize=(5, 3))
    ax_el.plot(range(1, 10), inertia, marker='o', color=WARNA['aktual'],
               linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2.5)
    ax_el.fill_between(range(1, 10), inertia, alpha=0.08, color=WARNA['aktual'])
    style_ax(ax_el, title='Metode Elbow — Penentuan Jumlah Cluster Optimal',
             xlabel='Jumlah Cluster', ylabel='Inertia')
    ax_el.grid(axis='x', linestyle='--', alpha=0.3)
    ax_el.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    col_elbow, _ = st.columns([1, 1])
    with col_elbow:
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
    for i, idx in enumerate(cluster_avg.index[:3]):
        mapping_cluster[idx] = ['Fast Moving', 'Medium Moving', 'Slow Moving'][i]

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

    st.subheader("Daftar Produk per Cluster")
    pilih_lihat = st.selectbox("Tampilkan produk dalam cluster:",
                               ['Fast Moving', 'Medium Moving', 'Slow Moving'],
                               key='lihat_cluster')
    produk_lihat = filtered_data[filtered_data['Kategori'] == pilih_lihat].index.tolist()
    df_lihat = pd.DataFrame({'Produk': produk_lihat})
    df_lihat.index = range(1, len(df_lihat) + 1)
    st.dataframe(df_lihat, use_container_width=True)

    st.session_state['filtered_data']   = filtered_data
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

    col_a, col_b = st.columns(2)
    with col_a:
        pilih_cluster  = st.selectbox("Pilih Cluster",
                                      ['Fast Moving', 'Medium Moving', 'Slow Moving'])
        produk_cluster = filtered_data[
            filtered_data['Kategori'] == pilih_cluster
        ].index.tolist()
        produk = st.selectbox("Pilih Produk", produk_cluster)

    with col_b:
        metode = st.selectbox(
            "Pilih Metode Forecasting",
            [
                "🤖 Otomatis (Rekomendasi Sistem)",
                "Holt-Winters Additive",
                "Holt-Winters Multiplicative",
                "ETS",
                "ARIMA",
            ]
        )
        jumlah_forecast = st.slider("Jumlah Bulan yang Diramalkan", 1, 12, 6)

    if metode == "🤖 Otomatis (Rekomendasi Sistem)":
        st.info("💡 Sistem akan menjalankan semua metode, membandingkan akurasinya, "
                "lalu otomatis menggunakan metode terbaik untuk forecast ke depan.")

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
                color=WARNA['aktual'], linewidth=2.5, marker='o', markersize=6,
                label='Data Aktual')
    ax_akt.axvspan(test.index[0], test.index[-1], alpha=0.08, color='orange',
                   label=f'Periode Test ({len(test)} bln)')
    ax_akt.axvline(x=test.index[0], color=WARNA['vline'], linestyle='--', linewidth=1.5)
    style_ax(ax_akt, title=f'Data Aktual Barang Keluar — Produk {produk}',
             ylabel='Jumlah Keluar')
    ax_akt.legend()
    plt.tight_layout()
    st.pyplot(fig_akt)
    plt.close(fig_akt)

    # ==========================================
    # HELPER — JALANKAN SEMUA METODE
    # ==========================================

    def jalankan_semua_metode(train, test, data_produk, jumlah_forecast):
        hasil_eval   = {}
        hasil_future = {}

        # HW Additive
        fc_hwa = ExponentialSmoothing(
            train, trend='add', seasonal='add', seasonal_periods=12
        ).fit().forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Additive'] = {
            'forecast': fc_hwa,
            'mae':  mean_absolute_error(test.values, fc_hwa.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwa.values)),
        }
        hasil_future['HW Additive'] = ExponentialSmoothing(
            data_produk, trend='add', seasonal='add', seasonal_periods=12
        ).fit().forecast(jumlah_forecast).clip(lower=0)

        # HW Multiplicative
        tr_nz = train.copy(); tr_nz[tr_nz <= 0] = 1
        fc_hwm = ExponentialSmoothing(
            tr_nz, trend='add', seasonal='mul', seasonal_periods=12
        ).fit().forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['HW Multiplicative'] = {
            'forecast': fc_hwm,
            'mae':  mean_absolute_error(test.values, fc_hwm.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_hwm.values)),
        }
        fd_nz = data_produk.copy(); fd_nz[fd_nz <= 0] = 1
        hasil_future['HW Multiplicative'] = ExponentialSmoothing(
            fd_nz, trend='add', seasonal='mul', seasonal_periods=12
        ).fit().forecast(jumlah_forecast).clip(lower=0)

        # ETS
        fc_ets = ETSModel(
            train, error="add", trend="add", seasonal="add", seasonal_periods=12
        ).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)
        hasil_eval['ETS'] = {
            'forecast': fc_ets,
            'mae':  mean_absolute_error(test.values, fc_ets.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_ets.values)),
        }
        hasil_future['ETS'] = ETSModel(
            data_produk, error="add", trend="add", seasonal="add", seasonal_periods=12
        ).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)

        # ARIMA
        fc_arima = ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=jumlah_forecast)
        hasil_eval['ARIMA'] = {
            'forecast': fc_arima,
            'mae':  mean_absolute_error(test.values, fc_arima.values),
            'rmse': np.sqrt(mean_squared_error(test.values, fc_arima.values)),
        }
        hasil_future['ARIMA'] = ARIMA(
            data_produk, order=(1, 1, 1)
        ).fit().forecast(steps=jumlah_forecast)

        return hasil_eval, hasil_future

    # ==========================================
    # HELPER — TAMPILKAN FORECAST TERBAIK
    # ==========================================

    def tampilkan_forecast_terbaik(nama_terbaik, fc_best, mae, rmse):

        st.subheader(f"🏆 Forecast ke Depan — Metode Terbaik: {nama_terbaik}")
        st.caption(f"Model di-retrain menggunakan seluruh {len(data_produk)} bulan data.")

        best_df = pd.DataFrame({
            'Periode':        fc_best.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(fc_best.values, 2),
        })
        best_df.index = range(1, len(best_df) + 1)
        st.dataframe(best_df, use_container_width=True)

        csv_best = best_df.to_csv().encode('utf-8')
        st.download_button(
            f"⬇️ Download Forecast ({nama_terbaik})",
            csv_best, f"forecast_{produk}_{nama_terbaik}.csv", 'text/csv'
        )

        fig_bst, ax_bst = plt.subplots(figsize=(12, 4.5))
        ax_bst.plot(data_produk.index, data_produk.values,
                    color=WARNA['aktual'], linewidth=2.5, marker='o', markersize=6,
                    label=f'Data Aktual ({len(data_produk)} bln)')
        ax_bst.plot(fc_best.index, fc_best.values,
                    color=WARNA[nama_terbaik], linewidth=3,
                    marker='o', markersize=9, linestyle='--',
                    label=f'Forecast — {nama_terbaik} (Terbaik)')
        ax_bst.axvline(x=fc_best.index[0], color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7, label='Awal Forecast')
        ax_bst.axvspan(fc_best.index[0], fc_best.index[-1], alpha=0.08,
                       color=WARNA[nama_terbaik])
        style_ax(ax_bst,
                 title=f'Forecast ke Depan — {nama_terbaik} — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_bst.legend()
        plt.tight_layout()
        st.pyplot(fig_bst)
        plt.close(fig_bst)

        st.markdown("---")
        st.subheader("📝 Ringkasan Hasil Analisis")

        st.success(
            f"**Produk:** {produk}  |  "
            f"**Cluster:** {pilih_cluster}  |  "
            f"**Metode Terbaik:** {nama_terbaik}  |  "
            f"**MAE:** {mae:.2f}  |  "
            f"**RMSE:** {rmse:.2f}"
        )

        st.markdown("**📅 Detail Forecast per Bulan:**")
        col_ring = st.columns(min(len(best_df), 6))
        for i, row in best_df.iterrows():
            with col_ring[(i - 1) % len(col_ring)]:
                st.metric(
                    label=row['Periode'],
                    value=f"{int(row['Hasil Forecast'])} unit"
                )

        st.caption("💡 Gunakan hasil forecast metode terbaik sebagai acuan perencanaan "
                   "pengadaan barang pada periode berikutnya.")

    # ==========================================
    # HELPER — TAMPILKAN PERBANDINGAN
    # ==========================================

    def tampilkan_perbandingan(hasil_eval, nama_terbaik):

        st.markdown("---")
        st.subheader("📊 Perbandingan Akurasi Semua Metode")
        st.caption("Baris hijau = metode terbaik. Ditampilkan sebagai referensi tambahan.")

        perbandingan = pd.DataFrame([
            {'Metode': m, 'MAE': round(v['mae'], 2), 'RMSE': round(v['rmse'], 2)}
            for m, v in hasil_eval.items()
        ])
        perbandingan.index = range(1, len(perbandingan) + 1)

        def highlight_best(row):
            return ['background-color: #d4edda; font-weight: bold'
                    if row['Metode'] == nama_terbaik else '' for _ in row]

        st.dataframe(
            perbandingan.style.apply(highlight_best, axis=1),
            use_container_width=True
        )

        metode_list = perbandingan['Metode'].tolist()
        warna_list  = [WARNA[m] for m in metode_list]
        alpha_list  = [1.0 if m == nama_terbaik else 0.4 for m in metode_list]

        fig_cmp, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(13, 4))

        for i, m in enumerate(metode_list):
            ax_mae.bar(m, perbandingan.loc[i+1, 'MAE'],
                       color=warna_list[i], alpha=alpha_list[i],
                       edgecolor='white', width=0.5)
            ax_rmse.bar(m, perbandingan.loc[i+1, 'RMSE'],
                        color=warna_list[i], alpha=alpha_list[i],
                        edgecolor='white', width=0.5)

        for rect, val in zip(ax_mae.patches, perbandingan['MAE']):
            idx = list(ax_mae.patches).index(rect)
            fw  = 'bold' if metode_list[idx] == nama_terbaik else 'normal'
            ax_mae.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + perbandingan['MAE'].max() * 0.02,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=10, fontweight=fw)

        for rect, val in zip(ax_rmse.patches, perbandingan['RMSE']):
            idx = list(ax_rmse.patches).index(rect)
            fw  = 'bold' if metode_list[idx] == nama_terbaik else 'normal'
            ax_rmse.text(rect.get_x() + rect.get_width() / 2,
                         rect.get_height() + perbandingan['RMSE'].max() * 0.02,
                         f'{val:.2f}', ha='center', va='bottom',
                         fontsize=10, fontweight=fw)

        style_ax(ax_mae, title='Perbandingan MAE', ylabel='MAE')
        ax_mae.set_ylim(0, perbandingan['MAE'].max() * 1.3)
        style_ax(ax_rmse, title='Perbandingan RMSE', ylabel='RMSE')
        ax_rmse.set_ylim(0, perbandingan['RMSE'].max() * 1.3)
        plt.suptitle('Evaluasi Akurasi Semua Metode', fontsize=13,
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.subheader("📉 Grafik Evaluasi Gabungan")
        st.caption("Metode terbaik ditampilkan paling menonjol — metode lain sebagai pembanding.")

        fig_ev, ax_ev = plt.subplots(figsize=(13, 5))

        ax_ev.plot(train.index, train.values, color=WARNA['train'],
                   linewidth=2.5, marker='o', markersize=5, label='Data Train')
        ax_ev.plot(test.index, test.values, color=WARNA['test'],
                   linewidth=2.5, marker='o', markersize=7, label='Data Test (Aktual)')

        for nm, v in hasil_eval.items():
            if nm != nama_terbaik:
                ax_ev.plot(
                    v['forecast'].index, v['forecast'].values,
                    color=WARNA[nm], linewidth=1.5,
                    marker='s', markersize=5,
                    linestyle='--', alpha=0.35,
                    label=nm
                )

        ax_ev.plot(
            hasil_eval[nama_terbaik]['forecast'].index,
            hasil_eval[nama_terbaik]['forecast'].values,
            color=WARNA[nama_terbaik], linewidth=3.5,
            marker='o', markersize=9,
            linestyle='-', zorder=5,
            label=f'★ {nama_terbaik} (Terbaik)'
        )

        ax_ev.axvline(x=test.index[0], color=WARNA['vline'],
                      linestyle='--', linewidth=1.5, label='Awal Test')
        ax_ev.axvspan(test.index[0], test.index[-1], alpha=0.06, color='orange')
        style_ax(ax_ev, title=f'Evaluasi Gabungan Semua Metode — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_ev.legend(loc='upper left', ncol=2)
        plt.tight_layout()
        st.pyplot(fig_ev)
        plt.close(fig_ev)

    # ==========================================
    # MODE OTOMATIS
    # ==========================================

    if metode == "🤖 Otomatis (Rekomendasi Sistem)":

        with st.spinner("⏳ Menjalankan semua metode dan memilih yang terbaik..."):
            hasil_eval, hasil_future = jalankan_semua_metode(
                train, test, data_produk, jumlah_forecast
            )

        nama_terbaik = min(hasil_eval, key=lambda m: hasil_eval[m]['mae'])
        mae_terbaik  = hasil_eval[nama_terbaik]['mae']
        rmse_terbaik = hasil_eval[nama_terbaik]['rmse']
        fc_best      = hasil_future[nama_terbaik]

        tampilkan_perbandingan(hasil_eval, nama_terbaik)

        st.success(f"✅ Berdasarkan evaluasi di atas, sistem memilih metode terbaik: "
                   f"**{nama_terbaik}** (MAE = {mae_terbaik:.2f}  |  RMSE = {rmse_terbaik:.2f})")

        tampilkan_forecast_terbaik(nama_terbaik, fc_best, mae_terbaik, rmse_terbaik)

    # ==========================================
    # MODE MANUAL — SATU METODE
    # ==========================================

    else:

        nama_map = {
            "Holt-Winters Additive":       "HW Additive",
            "Holt-Winters Multiplicative": "HW Multiplicative",
            "ETS":   "ETS",
            "ARIMA": "ARIMA",
        }
        nama = nama_map[metode]

        with st.spinner(f"⏳ Menjalankan {nama}..."):
            if nama == "HW Additive":
                fc_eval   = ExponentialSmoothing(train, trend='add', seasonal='add',
                             seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
                fc_future = ExponentialSmoothing(data_produk, trend='add', seasonal='add',
                             seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)

            elif nama == "HW Multiplicative":
                tr_nz = train.copy(); tr_nz[tr_nz <= 0] = 1
                fc_eval = ExponentialSmoothing(tr_nz, trend='add', seasonal='mul',
                           seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)
                fd_nz = data_produk.copy(); fd_nz[fd_nz <= 0] = 1
                fc_future = ExponentialSmoothing(fd_nz, trend='add', seasonal='mul',
                             seasonal_periods=12).fit().forecast(jumlah_forecast).clip(lower=0)

            elif nama == "ETS":
                fc_eval   = ETSModel(train, error="add", trend="add", seasonal="add",
                             seasonal_periods=12).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)
                fc_future = ETSModel(data_produk, error="add", trend="add", seasonal="add",
                             seasonal_periods=12).fit(disp=False).forecast(jumlah_forecast).clip(lower=0)

            elif nama == "ARIMA":
                fc_eval   = ARIMA(train, order=(1,1,1)).fit().forecast(steps=jumlah_forecast)
                fc_future = ARIMA(data_produk, order=(1,1,1)).fit().forecast(steps=jumlah_forecast)

        mae  = mean_absolute_error(test.values, fc_eval.values)
        rmse = np.sqrt(mean_squared_error(test.values, fc_eval.values))

        c1, c2 = st.columns(2)
        with c1: st.metric("MAE",  f"{mae:.2f}")
        with c2: st.metric("RMSE", f"{rmse:.2f}")

        eval_df = pd.DataFrame({
            'Periode':             fc_eval.index.strftime('%b-%Y'),
            'Forecast (Evaluasi)': np.round(fc_eval.values, 2),
            'Data Aktual (Test)':  np.round(test.values, 2),
        })
        eval_df.index = range(1, len(eval_df) + 1)

        st.subheader("📋 Perbandingan Forecast vs Aktual (Periode Evaluasi)")
        st.dataframe(eval_df, use_container_width=True)

        fig_e, ax_e = plt.subplots(figsize=(12, 4.5))
        ax_e.plot(train.index, train.values, color=WARNA['train'],
                  linewidth=2.5, marker='o', markersize=5, label='Data Train')
        ax_e.plot(test.index, test.values, color=WARNA['test'],
                  linewidth=2.5, marker='o', markersize=7, label='Data Test (Aktual)')
        ax_e.plot(fc_eval.index, fc_eval.values, color=WARNA[nama],
                  linewidth=3, marker='o', markersize=8, linestyle='-',
                  label=f'Forecast {nama}')
        ax_e.axvline(x=test.index[0], color=WARNA['vline'],
                     linestyle='--', linewidth=1.5, label='Awal Test')
        ax_e.axvspan(test.index[0], test.index[-1], alpha=0.06, color='orange')
        style_ax(ax_e, title=f'Evaluasi Model {nama} — Produk {produk}',
                 ylabel='Jumlah Keluar')
        ax_e.legend()
        plt.tight_layout()
        st.pyplot(fig_e)
        plt.close(fig_e)

        tampilkan_forecast_terbaik(nama, fc_future, mae, rmse)
