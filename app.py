import streamlit as st
import pandas as pd
import pickle
import numpy as np

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
    return model_km, model_gmm, sc

kmeans, gmm, scaler = load_assets()

st.title("🎓 Klasifikasi Performa Akademik Siswa")
st.write("Aplikasi ini mengelompokkan siswa berdasarkan nilai akademik menggunakan algoritma Clustering.")

# Sidebar untuk input
st.sidebar.header("Input Nilai Siswa")
math_score = st.sidebar.slider("Math Score", 0, 100, 70)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 70)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 70)

algo_choice = st.selectbox("Pilih Algoritma Clustering", ["K-Means", "Gaussian Mixture Model (GMM)"])

if st.button("Analisis Cluster"):
    # Preprocessing input
    data = np.array([[math_score, reading_score, writing_score]])
    data_scaled = scaler.transform(data)
    
    # Prediksi
    if algo_choice == "K-Means":
        cluster = kmeans.predict(data_scaled)[0]
    else:
        cluster = gmm.predict(data_scaled)[0]
    
    # Tampilkan Hasil
    st.subheader(f"Hasil Prediksi: Cluster {cluster}")
    
    if cluster == 0:
        st.info("Karakteristik Cluster 0: Siswa dengan performa akademik yang cenderung sedang/rendah.")
    else:
        st.success("Karakteristik Cluster 1: Siswa dengan performa akademik yang tinggi.")

    # Visualisasi sederhana posisi data
    df_input = pd.DataFrame(data, columns=['Math', 'Reading', 'Writing'])
    st.table(df_input)
