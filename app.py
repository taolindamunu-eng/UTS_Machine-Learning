import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# Set judul halaman
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

@st.cache_resource
def load_assets():
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
st.write("Aplikasi ini mengelompokkan siswa berdasarkan nilai akademik secara konsisten di semua algoritma.")

# Sidebar untuk input
st.sidebar.header("Input Nilai Siswa")
math_score = st.sidebar.slider("Math Score", 0, 100, 70)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 70)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 70)

algo_choice = st.selectbox(
    "Pilih Algoritma Clustering",
    ["K-Means", "Gaussian Mixture Model (GMM)", "Agglomerative Clustering"]
)

if st.button("Analisis Cluster"):
    data = np.array([[math_score, reading_score, writing_score]])
    data_scaled = scaler.transform(data)
    raw_cluster = None

    # 1. Prediksi berdasarkan algoritma yang dipilih
    if algo_choice == "K-Means":
        raw_cluster = kmeans.predict(data_scaled)[0]

    elif algo_choice == "Gaussian Mixture Model (GMM)":
        raw_cluster = gmm.predict(data_scaled)[0]

    elif algo_choice == "Agglomerative Clustering":
        if agglo_centroids is not None:
            distances = cdist(data_scaled, agglo_centroids)
            raw_cluster = np.argmin(distances)
        else:
            st.error("Centroid Agglomerative tidak ditemukan.")

    # 2. EKSEKUSI PENUKARAN LABEL UNTUK SEMUA ALGORITMA
    if raw_cluster is not None:
        # Berdasarkan temuan kita, model aslimu memberikan label 0 untuk skor tinggi (di gambar awal)
        # Namun kamu ingin standar baru: Cluster 0 = TINGGI, Cluster 1 = RENDAH.
        # Jika hasil prediksi mentah (raw) adalah 0 (rendah), kita balik jadi 1.
        # Jika hasil prediksi mentah (raw) adalah 1 (tinggi), kita balik jadi 0.
        
        final_cluster = 1 if raw_cluster == 0 else 0

        # 3. Tampilkan hasil dengan label yang sudah konsisten
        st.subheader(f"Hasil Prediksi ({algo_choice}): Cluster {final_cluster}")

        if final_cluster == 0:
            st.success("✨ **Karakteristik Cluster 0: Performa Akademik Tinggi.**")
            st.markdown("- Penguasaan materi sangat baik.\n- Nilai di atas rata-rata keseluruhan.")
        else:
            st.warning("⚠️ **Karakteristik Cluster 1: Performa Akademik Rendah/Sedang.**")
            st.markdown("- Memerlukan bimbingan tambahan.\n- Fokus pada peningkatan nilai dasar.")

    # Tabel input untuk konfirmasi user
    st.table(pd.DataFrame(data, columns=['Math', 'Reading', 'Writing']))
