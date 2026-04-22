import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# Set judul halaman
st.set_page_config(page_title="Student Performance Prediction")

# Load model dan scaler
@st.cache_resource
def load_assets():
    with open('model_kmeans.pkl', 'rb') as f:
        model_km = pickle.load(f)
    with open('model_gmm.pkl', 'rb') as f:
        model_gmm = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    
    # centroid untuk agglomerative (WAJIB ADA)
    try:
        with open('agglo_centroids.pkl', 'rb') as f:
            agglo_centroids = pickle.load(f)
    except:
        agglo_centroids = None

    return model_km, model_gmm, sc, agglo_centroids

kmeans, gmm, scaler, agglo_centroids = load_assets()

st.title("🎓 Klasifikasi Performa Akademik Siswa")
st.write("Aplikasi ini mengelompokkan siswa berdasarkan nilai akademik menggunakan algoritma Clustering.")

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
    # Preprocessing input
    data = np.array([[math_score, reading_score, writing_score]])
    data_scaled = scaler.transform(data)

    # Prediksi
    if algo_choice == "K-Means":
        cluster = kmeans.predict(data_scaled)[0]

    elif algo_choice == "Gaussian Mixture Model (GMM)":
        cluster = gmm.predict(data_scaled)[0]

    elif algo_choice == "Agglomerative Clustering":
        if agglo_centroids is None:
            st.error("File agglo_centroids.pkl tidak ditemukan. Silakan buat terlebih dahulu saat training.")
            cluster = None
        else:
            distances = cdist(data_scaled, agglo_centroids)
            cluster = np.argmin(distances)

    # Tampilkan hasil
    if cluster is not None:
        st.subheader(f"Hasil Prediksi: Cluster {cluster}")

        if cluster == 0:
            st.info("Karakteristik Cluster 0: Performa cenderung rendah/sedang.")
        elif cluster == 1:
            st.success("Karakteristik Cluster 1: Performa tinggi.")
        else:
            st.warning("Cluster tambahan terdeteksi.")

    # Info khusus Agglomerative
    if algo_choice == "Agglomerative Clustering":
        st.info("Agglomerative Clustering menggunakan pendekatan jarak ke centroid karena tidak memiliki fungsi predict bawaan.")

    # Tampilkan data input
    df_input = pd.DataFrame(data, columns=['Math', 'Reading', 'Writing'])
    st.table(df_input)
