import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from linearmodels.panel import PooledOLS, RandomEffects
from PIL import Image
import streamlit.components.v1 as components
from statsmodels.api import add_constant
import base64
from io import BytesIO

# Load gambar
logo_uns = Image.open("logouns.png")
logo_magang = Image.open("SOLUSI247.png")

#Logo Kampus dan Logo Intansi Magang
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

b64_uns = image_to_base64(logo_uns)
b64_magang = image_to_base64(logo_magang)

html = f"""
<div style='display: flex; justify-content: center; gap: 20px;'>
    <img src='data:image/png;base64,{b64_uns}' width='80'/>
    <img src='data:image/png;base64,{b64_magang}' width='80'/>
</div>
"""

st.sidebar.markdown(html, unsafe_allow_html=True)

# Layout halaman
st.set_page_config(page_title="Regresi Panel Aset", layout="wide")
st.title("📈 Analisis Pengaruh Berita Ekonomi Amerika Serikat terhadap Kelas Aset menggunakan Regresi Data Panel")
st.sidebar.markdown("### 👨‍💻 Disusun oleh:")
st.sidebar.markdown("""
**Andi Ulia Rahman**  
NIM: `M0722011`  

**Putri**  
NIM: `M0722063`  
""")

# Load data untuk regresi data panel dari Excel
@st.cache_data
def load_data():
    df = pd.read_excel("data/Data_Streamlit.xlsx")  # Pastikan file ada di folder data/
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df_panel = load_data()
# Bersihkan dan siapkan data panel
df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')
df_panel = df_panel.dropna(subset=['Date', 'Kelas_Aset', 'Return', 'Surprise'])

# Set MultiIndex
df = df_panel.set_index(['Kelas_Aset', 'Date'])
y = df['Return']
X = df[['Surprise']]

X_pooled = add_constant(X)

# === 1. POOL OLS ===
pooled_model = PooledOLS(y, X_pooled).fit()

# === 2. FIXED EFFECTS (LSDV) ===
df_reset = df_panel[['Kelas_Aset', 'Date', 'Return', 'Surprise']].copy()
aset_dummies = pd.get_dummies(df_reset['Kelas_Aset'], drop_first=True, prefix='Kelas_Aset')
X_lsdv = pd.concat([df_reset[['Surprise']], aset_dummies], axis=1)
X_lsdv = add_constant(X_lsdv)
X_lsdv.index = pd.MultiIndex.from_frame(df_reset[['Kelas_Aset', 'Date']])
y_lsdv = df_reset['Return']
y_lsdv.index = pd.MultiIndex.from_frame(df_reset[['Kelas_Aset', 'Date']])
lsdv_model = PooledOLS(y_lsdv, X_lsdv).fit()

# === 3. RANDOM EFFECTS ===
random_model = RandomEffects(y, X_pooled).fit()


# Tab navigasi
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📘 Deskripsi Proyek", "📊 Eksplorasi Data", "📈 Regresi Data Panel", "✅ Uji Asumsi", "📝 Kesimpulan"])

# === TAB 1: Deskripsi Proyek ===
with tab1:
    st.subheader("Latar Belakang")
    st.markdown("""
    Pergerakan harga aset keuangan sangat dipengaruhi oleh kejutan dalam rilis berita ekonomi penting, terutama dari Amerika Serikat.
    Proyek ini bertujuan untuk mengukur pengaruh kejutan berita (surprise) terhadap return berbagai kelas aset menggunakan model regresi data panel.
    Digunakan model Pooled OLS, Fixed Effects, dan Random Effects untuk menentukan pendekatan terbaik.
    """)

# === TAB 2: Eksplorasi Data ===
with tab2:
    st.subheader("📊 Tabel Data Panel")
    st.dataframe(df_panel)

    st.subheader("Distribusi Return")
    fig, ax = plt.subplots()
    sns.histplot(df_panel['Return'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Rata-rata Return per Aset")
    avg_return = df_panel.groupby("Kelas_Aset")["Return"].mean()
    st.bar_chart(avg_return)
    
    # Pastikan Date format datetime
    df_panel['Date'] = pd.to_datetime(df_panel['Date'])
    # Sort data per aset dan tanggal
    df_panel = df_panel.sort_values(by=['Kelas_Aset', 'Date'])

    # Tambahkan urutan rilis berita per aset
    df_panel['Berita_ke'] = df_panel.groupby('Kelas_Aset').cumcount() + 1

    st.subheader("📈 Respon Return Aset terhadap Setiap Rilis Berita ADP")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_panel, x="Berita_ke", y="Return", hue="Kelas_Aset", marker="o", ax=ax)

    ax.set_xlabel("Rilis Berita ke-")
    ax.set_ylabel("Return (%)")
    ax.set_title("Respon Return Kelas Aset terhadap Kejutan Berita ADP (Event ke-1, 2, 3)")
    ax.set_xticks([1, 2, 3])
    st.pyplot(fig)
    
    st.subheader("📌 Hubungan Surprise Berita dengan Return Aset")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_panel, x="Surprise", y="Return", hue="Kelas_Aset", ax=ax5)
    ax5.set_title("Scatterplot: Surprise vs Return")
    ax5.set_xlabel("Surprise (Kejutan Berita ADP)")
    ax5.set_ylabel("Return (%)")
    st.pyplot(fig5)



# === TAB 3: Regresi Data Panel ===
with tab3:
    st.subheader("Hasil Regresi Data Panel")

    # --- Tampilkan koefisien, p-value, dan R Square dari ketiganya ---
    st.markdown("### 📌 Ringkasan Koefisien dan P-value")

    def extract_summary(model, label):
        df_result = pd.DataFrame({
            "Koefisien": model.params,
            "P-Value": model.pvalues
        })
        df_result.index.name = "Variabel"
        st.subheader(label)
        st.dataframe(df_result)

    extract_summary(pooled_model, "Model 1: Pooled OLS")
    extract_summary(lsdv_model, "Model 2: LSDV (Fixed Effects via Dummy)")
    extract_summary(random_model, "Model 3: Random Effects")

    st.subheader("🧪 Pemilihan Model Terbaik")
    st.markdown("""
    Uji Chow digunakan untuk menentukan apakah model Fixed Effects lebih baik dibandingkan model Pooled OLS.
    Jika hasil uji Chow signifikan, maka model Fixed Effects lebih disarankan karena mengakomodasi efek individual masing-masing aset.
    """)

       # --- UJI CHOW MANUAL ---
    st.markdown("### 🧮 Uji Chow Manual: Pooled OLS vs Fixed Effects (LSDV)")

    from scipy import stats

    # Hitung RSS (Residual Sum of Squares)
    rss_pooled = (pooled_model.resids ** 2).sum()
    rss_lsdv = (lsdv_model.resids ** 2).sum()

    n = len(y)  # jumlah observasi
    k_pooled = X_pooled.shape[1]
    k_lsdv = X_lsdv.shape[1]

    # Hitung F-statistik
    f_stat = ((rss_pooled - rss_lsdv) / (k_lsdv - k_pooled)) / (rss_lsdv / (n - k_lsdv))
    p_value_chow = 1 - stats.f.cdf(f_stat, k_lsdv - k_pooled, n - k_lsdv)

    # Tampilkan hasil
    st.write(f"**F-Statistic:** {f_stat:.4f}")
    st.write(f"**P-Value:** {p_value_chow:.4f}")
    st.write(f"RSS Pooled OLS: {rss_pooled:.4f}")
    st.write(f"RSS LSDV (Fixed Effects): {rss_lsdv:.4f}")

    # Interpretasi
    if p_value_chow < 0.05:
        st.success("🟢 Hasil: Tolak H₀ → Model LSDV (Fixed Effects) **lebih baik** dari Pooled OLS.")
    else:
        st.info("🔵 Hasil: Gagal Tolak H₀ → Model Pooled OLS **Sudah Cukup**, tidak perlu LSDV.")
        st.markdown("### ✅ Penentuan Model Terbaik")

    st.markdown("""
    Berdasarkan hasil **Uji Chow**, diperoleh bahwa model **Pooled OLS** secara statistik lebih tepat dibandingkan model Fixed Effects (LSDV), 
    ditunjukkan oleh nilai F-statistik yang signifikan.

    Karena model **Pooled OLS telah terpilih sebagai model terbaik**, maka **uji Hausman (untuk membandingkan Fixed Effects vs Random Effects) tidak diperlukan**. 
    Uji Hausman hanya relevan jika Fixed Effects lebih unggul dari Pooled OLS.
    """)

    st.markdown("### 🧾 Persamaan Regresi Terbaik: Pooled OLS")

    st.latex(r"""
        \text{Return}_{it} = \beta_0 + \beta_1 \cdot \text{Surprise}_{it}
    """)

    st.markdown(f"""
    Berdasarkan hasil estimasi:

    - Intercept (β₀): **{pooled_model.params['const']:.4f}**
    - Koefisien Surprise (β₁): **{pooled_model.params['Surprise']:.4f}**

    Maka persamaan regresinya adalah:
    """)

    st.latex(fr"""
        \text{{Return}}_{{it}} = {pooled_model.params['const']:.4f} \,-\, {abs(pooled_model.params['Surprise']):.4f} \cdot \text{{Surprise}}_{{it}}
    """)

    st.markdown("📌 **Interpretasi:** Jika nilai surprise naik 1 unit, return akan turun sebesar {:.2f}%".format(abs(pooled_model.params['Surprise'])*100))

# === TAB 4: Uji Asumsi ===
with tab4:
    st.subheader("✅ Uji Asumsi Regresi Panel (Pooled OLS)")

    # --- Preprocessing untuk model PooledOLS ---
    df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')
    df_panel = df_panel.dropna(subset=['Date', 'Kelas_Aset', 'Return', 'Surprise'])
    df = df_panel.set_index(['Kelas_Aset', 'Date'])

    y = df['Return']
    X = df[['Surprise']]
    X = add_constant(X)

    model = PooledOLS(y, X).fit()
    residuals = model.resids

    # === UJI HETEROSKEDASTISITAS (Breusch-Pagan)
    st.markdown("### 🔎 Uji Heteroskedastisitas (Breusch-Pagan)")
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(residuals, X)
    bp_labels = ['LM Statistic', 'LM p-value', 'F-statistic', 'F p-value']
    bp_result = dict(zip(bp_labels, bp_test))
    st.write(bp_result)

    if bp_result['F p-value'] < 0.05:
        st.error("Terdeteksi heteroskedastisitas ❌")
    else:
        st.success("Tidak terdeteksi heteroskedastisitas ✅")

    # === UJI NORMALITAS (Jarque-Bera)
    st.markdown("### 🔎 Uji Normalitas Residual (Jarque-Bera)")
    from scipy.stats import jarque_bera
    jb_stat, jb_pval = jarque_bera(residuals)
    st.write({"JB Statistic": jb_stat, "p-value": jb_pval})

    if jb_pval < 0.05:
        st.error("Distribusi residual tidak normal ❌")
    else:
        st.success("Distribusi residual normal ✅")
    
    # === UJI AUTOKORELASI (Ljung-Box)
    st.markdown("### 🔎 Uji Autokorelasi Residual (Ljung-Box Test)")

    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_stat = lb_result['lb_stat'].values[0]
    lb_pval = lb_result['lb_pvalue'].values[0]
    st.write({"Ljung-Box Statistic (lag 10)": lb_stat, "p-value": lb_pval})

    if lb_pval < 0.05:
        st.error("Terdeteksi autokorelasi pada residual ❌")
    else:
        st.success("Tidak terdeteksi autokorelasi pada residual ✅")


with tab5:
    st.header("📝 Kesimpulan")
    st.markdown("""
    Berdasarkan hasil analisis regresi OLS yang dilakukan terhadap pengaruh *Surprise* dari data ADP Nonfarm Employment terhadap return kelas aset, diperoleh kesimpulan sebagai berikut:

    - Variabel **Surprise ADP Nonfarm Employment** memiliki **pengaruh negatif dan signifikan** terhadap return kelas aset. Setiap kenaikan 1 poin dalam nilai surprise menyebabkan penurunan return rata-rata sebesar **1,74%** (*p-value = 0,009*).
    - Secara keseluruhan, model regresi yang dibangun signifikan, dibuktikan oleh nilai **F-statistic sebesar 7,947** dengan *p-value = 0,0087*.
    - Nilai **R-squared sebesar 22,1%** menunjukkan bahwa variasi return dapat dijelaskan oleh variabel Surprise ADP dalam proporsi yang cukup, meskipun tidak dominan.
    - Hasil ini memberikan bukti bahwa **kejutan ekonomi dari rilis data ADP Nonfarm Employment berdampak terhadap pasar**, khususnya pada pergerakan return berbagai kelas aset.
    - Temuan ini mengindikasikan bahwa pelaku pasar bereaksi negatif terhadap deviasi antara ekspektasi dan realisasi data ADP, dan menjadikan data ini sebagai indikator penting dalam strategi pengambilan keputusan investasi.

    Secara umum, hasil analisis mendukung hipotesis bahwa berita ekonomi, khususnya surprise dari data indikator ketenagakerjaan memiliki peran signifikan dalam mempengaruhi kinerja aset, dan penting untuk diperhatikan oleh investor maupun analis pasar.
    """)
