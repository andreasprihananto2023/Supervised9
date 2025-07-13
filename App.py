import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Memuat model Random Forest yang telah dilatih
with open('best_rf_model.pkl', 'rb') as model_file:
    best_rf = pickle.load(model_file)

# Membangun aplikasi Streamlit
def predict_delay():
    st.title("Prediksi Keterlambatan Pengiriman")

    # Input untuk data baru
    pizza_complexity = st.slider("Pizza Complexity", 1, 5, 3)
    order_hour = st.slider("Order Hour (Jam Pemesanan)", 0, 23, 14)
    restaurant_avg_time = st.slider("Restaurant Average Time (Menit)", 10, 60, 25)
    estimated_duration = st.slider("Estimated Duration (Menit)", 10, 60, 30)
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
        'Estimated Duration (min)': [estimated_duration],
        'Distance (km)': [distance],
        'Topping Density': [topping_density],
        'Traffic Level': [traffic_level],
        'Is Peak Hour': [is_peak_hour],
        'Is Weekend': [is_weekend]
    })

    # Prediksi delay menggunakan model
    predicted_delay = best_rf.predict(new_data)
    st.write(f"Predicted Delay (min): {predicted_delay[0]:.2f} minutes")

    if predicted_delay[0] > 0:
        st.write(f"Akan terjadi keterlambatan sebesar {predicted_delay[0]:.2f} menit.")
    else:
        st.write("Tidak ada keterlambatan.")

if __name__ == "__main__":
    predict_delay()
