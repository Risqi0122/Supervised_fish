import streamlit as st
import pickle
import numpy as np
import os

# Memuat model Perceptron
with open('perceptron_fish.pkl', 'rb') as model_file:
    model_perceptron = pickle.load(model_file)

# Memuat model Logistic Regression
if os.path.exists('Logistic_Regression_fruit_model.pkl'):
    with open('Logistic_Regression_fruit_model.pkl', 'rb') as file:
        model_Logistic_Regression = pickle.load(file)
else:
    st.error("File model Logistic Regression tidak ditemukan!")

# Judul Aplikasi
st.title('Prediksi Spesies Ikan')

# Dropdown untuk memilih model
model_choice = st.selectbox(
    'Pilih Model untuk Prediksi:',
    ('perceptron', 'LR')
)

# Input untuk setiap fitur ikan
length = st.number_input('Panjang Ikan (length):', min_value=0.0)
weight = st.number_input('Berat Ikan (weight):', min_value=0.0)
w_l_ratio = st.number_input('Rasio Berat ke Panjang (w_l_ratio):', min_value=0.0)

# Tombol untuk memprediksi spesies ikan
if st.button('Prediksi Spesies'):
    features = np.array([[length, weight, w_l_ratio]])
    
    # Memilih model berdasarkan pilihan pengguna
    if model_choice == 'perceptron':
        model = model_perceptron
        with open('label_encoder_fish_Perseptron.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
    else:  # For 'LR' (Logistic Regression)
        model = model_Logistic_Regression
        with open('LabelEncoder_fruit.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
    
    # Prediksi spesies ikan
    species = model.predict(features)[0]
    species_name = encoder.inverse_transform([species])[0]
    
    st.success(f'Spesies yang Diprediksi: {species_name}')
