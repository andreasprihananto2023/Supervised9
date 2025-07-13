import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Memuat model Random Forest yang telah dilatih
with open('best_rf_model.pkl', 'rb') as model_file:
    best_rf = pickle.load(model_file)

# Definisikan kolom fitur yang sama dengan X_train di pelatihan
feature_columns = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                   'Distance (km)', 'Topping Density', 'Traffic Level', 
                   'Is Peak Hour', 'Is Weekend']

# Membangun aplikasi Streamlit
def predict_duration():
    st.title("Prediksi Estimated Duration Pengiriman")

    # Input untuk data baru
    pizza_complexity = st.slider("Pizza Complexity", 1, 5, 3)
    order_hour = st.slider("Order Hour (Jam Pemesanan)", 0, 23, 14)
    restaurant_avg_time = st.slider("Restaurant Average Time (Menit)", 10, 60, 25)
    distance = st.slider("Distance (km)", 1, 10, 5)
    topping_density = st.slider("Topping Density", 1, 5, 2)
    traffic_level = st.slider("Traffic Level", 1, 5, 3)
    is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1], index=1)  # Peak hour (1 for yes, 0 for no)
    is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0)  # Weekend (1 for yes, 0 for no)

    # Data baru untuk prediksi
    new_data = pd.DataFrame({
        'Pizza Complexity': [pizza_complexity],
        'Order Hour': [order_hour],
        'Restaurant Avg Time': [restaurant_avg_time],
        'Distance (km)': [distance],
        'Topping Density': [topping_density],
        'Traffic Level': [traffic_level],
        'Is Peak Hour': [is_peak_hour],
        'Is Weekend': [is_weekend]
    })

    # Menyusun ulang kolom pada new_data sesuai dengan urutan feature_columns
    new_data = new_data[feature_columns]

    # Prediksi Estimated Duration menggunakan model
    predicted_duration = best_rf.predict(new_data)
    st.write(f"Predicted Estimated Duration (min): {predicted_duration[0]:.2f} minutes")

    # Memberi tahu pengguna jika estimasi waktu pengiriman lebih cepat atau terlambat
    if predicted_duration[0] > 30:  # Jika estimasi lebih besar dari 30 menit, anggap itu terlambat
        st.write(f"Estimasi pengiriman seharusnya lebih lama. Prediksi Estimated Duration adalah {predicted_duration[0]:.2f} menit.")
    else:
        st.write(f"Estimasi pengiriman lebih cepat dari yang diharapkan.")

if __name__ == "__main__":
    predict_duration()
