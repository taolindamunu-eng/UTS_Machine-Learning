import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cdist

# Set judul halaman
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

@st.cache_resource
def load_assets():
    # Memuat model yang sudah disinkronkan (0 = Tinggi)
    with open('model_kmeans.pkl', 'rb') as f:
        model_km = pickle.load(f)
    with open('model_gmm.pkl', 'rb') as f:
        model_gmm = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    
    try:
        with open('agglo_centroids.pkl', 'rb') as f:
            agglo_centroids = pickle.load(f)
    except:
        agglo_centroids = None

    return model_km, model_gmm, sc, agglo_centroids

kmeans, gmm, scaler, agglo_centroids = load_assets()

st.title("🎓 Klasifikasi Performa Akademik Siswa")
st.write("Aplikasi ini menggunakan model yang telah disinkronkan: **Cluster 0 selalu berarti Performa Tinggi**.")

# Sidebar untuk input
st.sidebar.header("Input Nilai Siswa")
math_score = st.sidebar.slider("Math Score", 0, 100, 85)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 85)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 85)

algo_choice = st.selectbox(
    "Pilih Algoritma Clustering",
    ["K-Means", "Gaussian Mixture Model (GMM)", "Agglomerative Clustering"]
)

if st.button("Analisis Cluster"):
    # 1. Preprocessing (Input harus di-scale agar sesuai dengan saat training)
    data = np.array([[math_score, reading_score, writing_score]])
    data_scaled = scaler.transform(data)
    cluster = None

    # 2. Prediksi Langsung (Tanpa perlu tukar label lagi di sini)
    if algo_choice == "K-Means":
        cluster = kmeans.predict(data_scaled)[0]

    elif algo_choice == "Gaussian Mixture Model (GMM)":
        cluster = gmm.predict(data_scaled)[0]

    elif algo_choice == "Agglomerative Clustering":
        if agglo_centroids is not None:
            # Menghitung jarak ke centroid yang sudah urut (index 0 = Tinggi)
            distances = cdist(data_scaled, agglo_centroids)
            cluster = np.argmin(distances)
        else:
            st.error("Centroid Agglomerative tidak ditemukan.")

    # 3. Tampilkan Hasil
    if cluster is not None:
        st.subheader(f"Hasil Prediksi ({algo_choice}): Cluster {cluster}")

        if cluster == 0:
            st.success("✨ **Karakteristik Cluster 0: Performa Akademik Tinggi.**")
            st.markdown("- Penguasaan materi sangat baik.\n- Nilai berada di kelompok atas.")
        else:
            st.warning("⚠️ **Karakteristik Cluster 1: Performa Akademik Rendah/Sedang.**")
            st.markdown("- Memerlukan bimbingan tambahan.\n- Fokus pada peningkatan nilai dasar.")

    # Tabel input untuk konfirmasi
    st.table(pd.DataFrame(data, columns=['Math', 'Reading', 'Writing']))