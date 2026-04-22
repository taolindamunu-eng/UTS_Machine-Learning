import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cdist

# Set judul halaman agar terlihat profesional
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

# Memuat aset model dengan caching agar aplikasi cepat
@st.cache_resource
def load_assets():
    try:
        with open('model_kmeans.pkl', 'rb') as f:
            model_km = pickle.load(f)
        with open('model_gmm.pkl', 'rb') as f:
            model_gmm = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        with open('agglo_centroids.pkl', 'rb') as f:
            agglo_centroids = pickle.load(f)
        return model_km, model_gmm, sc, agglo_centroids
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file: {e}. Pastikan semua file .pkl sudah tersedia.")
        return None, None, None, None

kmeans, gmm, scaler, agglo_centroids = load_assets()

# --- UI Header ---
st.title("🎓 Klasifikasi Performa Akademik Siswa")
st.markdown("""
Aplikasi ini mengelompokkan siswa berdasarkan skor akademik menggunakan model yang telah divalidasi.
**Standar Label:**
* **Cluster 0**: Performa Tinggi (High Performance)
* **Cluster 1**: Performa Rendah/Sedang (Lower Performance)
""")

# --- Sidebar Input ---
st.sidebar.header("📝 Input Nilai Siswa")
math_score = st.sidebar.slider("Math Score", 0, 100, 80)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 80)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 80)

algo_choice = st.selectbox(
    "Pilih Algoritma Clustering",
    ["K-Means", "Gaussian Mixture Model (GMM)", "Agglomerative Clustering"]
)

# --- Logika Analisis ---
if st.button("Jalankan Analisis"):
    if scaler is not None:
        # 1. Transformasi Data (Sangat krusial untuk ML)
        data = np.array([[math_score, reading_score, writing_score]])
        data_scaled = scaler.transform(data)
        cluster = None

        # 2. Prediksi berdasarkan model yang dipilih
        if algo_choice == "K-Means":
            cluster = kmeans.predict(data_scaled)[0]

        elif algo_choice == "Gaussian Mixture Model (GMM)":
            cluster = gmm.predict(data_scaled)[0]

        elif algo_choice == "Agglomerative Clustering":
            if agglo_centroids is not None:
                # Menggunakan jarak Euclidean ke centroid hasil profiling
                distances = cdist(data_scaled, agglo_centroids)
                cluster = np.argmin(distances)
            else:
                st.error("Centroid Agglomerative tidak ditemukan.")

        # 3. Output Visual
        if cluster is not None:
            st.divider()
            st.subheader(f"Hasil Prediksi: Cluster {cluster}")
            
            if cluster == 0:
                st.success("### ✨ Karakteristik: Performa Tinggi")
                st.write(f"Berdasarkan algoritma **{algo_choice}**, siswa ini masuk ke dalam kelompok dengan rata-rata nilai akademik yang superior.")
                st.markdown("- **Kekuatan:** Konsistensi nilai di atas 70.\n- **Rekomendasi:** Berikan materi pengayaan atau akselerasi.")
            else:
                st.warning("### ⚠️ Karakteristik: Performa Rendah/Sedang")
                st.write(f"Berdasarkan algoritma **{algo_choice}**, siswa ini masuk ke dalam kelompok yang memerlukan perhatian lebih.")
                st.markdown("- **Kelemahan:** Nilai cenderung di bawah 60.\n- **Rekomendasi:** Perlu bimbingan belajar tambahan atau mentoring sebaya.")

            # Menampilkan tabel data input untuk verifikasi
            st.table(pd.DataFrame(data, columns=['Math', 'Reading', 'Writing']))
    else:
        st.error("Aplikasi tidak dapat dijalankan karena model belum dimuat.")