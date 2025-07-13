import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Memuat model Random Forest yang telah dilatih
with open('best_rf_model.pkl', 'rb') as model_file:
    model_info = pickle.load(model_file)
    
# Ekstrak model dan informasi fitur
if isinstance(model_info, dict):
    best_rf = model_info['model']
    features = model_info['features']
else:
    # Untuk backward compatibility jika model lama
    best_rf = model_info
    features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
               'Distance (km)', 'Topping Density', 'Traffic Level', 
               'Is Peak Hour', 'Is Weekend']

# Membangun aplikasi Streamlit
def predict_estimated_duration():
    st.title("Prediksi Estimasi Durasi Pengiriman")
    
    st.markdown("""
    Aplikasi ini memprediksi estimasi durasi pengiriman berdasarkan berbagai faktor
    seperti kompleksitas pizza, jam pemesanan, jarak, dan kondisi lalu lintas.
    """)

    # Input untuk data baru
    col1, col2 = st.columns(2)
    
    with col1:
        pizza_complexity = st.slider("Pizza Complexity", 1, 5, 3, 
                                   help="Tingkat kompleksitas pizza (1=sederhana, 5=kompleks)")
        order_hour = st.slider("Order Hour (Jam Pemesanan)", 0, 23, 14,
                             help="Jam pemesanan dalam format 24 jam")
        restaurant_avg_time = st.slider("Restaurant Average Time (Menit)", 10, 60, 25,
                                       help="Rata-rata waktu persiapan restoran")
        distance = st.slider("Distance (km)", 1, 10, 5,
                           help="Jarak pengiriman dalam kilometer")
    
    with col2:
        topping_density = st.slider("Topping Density", 1, 5, 2,
                                  help="Kepadatan topping (1=sedikit, 5=banyak)")
        traffic_level = st.slider("Traffic Level", 1, 5, 3,
                                help="Tingkat kemacetan lalu lintas (1=lancar, 5=macet)")
        is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1], index=1,
                                  help="Apakah waktu pemesanan dalam jam sibuk? (11-14 atau 17-20)")
        is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0,
                                help="Apakah hari weekend? (Berdasarkan bulan 6,7,8,9)")

    # Data baru untuk prediksi - pastikan urutan dan nama kolom sesuai dengan training
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
    
    # Pastikan urutan kolom sesuai dengan yang digunakan saat training
    new_data = new_data[features]

    # Tombol untuk prediksi
    if st.button("Prediksi Estimasi Durasi", type="primary"):
        # Prediksi estimated duration menggunakan model
        predicted_duration = best_rf.predict(new_data)
        
        st.success(f"**Estimasi Durasi Pengiriman: {predicted_duration[0]:.2f} menit**")
        
        # Kategorisasi berdasarkan durasi
        if predicted_duration[0] <= 30:
            st.info("🟢 Pengiriman Cepat - Estimasi waktu sangat baik!")
        elif predicted_duration[0] <= 45:
            st.warning("🟡 Pengiriman Normal - Estimasi waktu dalam batas wajar")
        else:
            st.error("🔴 Pengiriman Lambat - Estimasi waktu cukup lama")
        
        # Konversi ke jam dan menit untuk tampilan yang lebih user-friendly
        hours = int(predicted_duration[0] // 60)
        minutes = int(predicted_duration[0] % 60)
        
        if hours > 0:
            st.write(f"📅 Estimasi waktu: {hours} jam {minutes} menit")
        else:
            st.write(f"📅 Estimasi waktu: {minutes} menit")
    
    # Informasi tambahan
    st.markdown("---")
    st.markdown("### 📊 Informasi Fitur")
    st.markdown("""
    - **Pizza Complexity**: Tingkat kesulitan pembuatan pizza
    - **Order Hour**: Jam pemesanan (mempengaruhi beban kerja)
    - **Restaurant Avg Time**: Rata-rata waktu persiapan restoran
    - **Distance**: Jarak pengiriman
    - **Topping Density**: Kepadatan topping pada pizza
    - **Traffic Level**: Kondisi lalu lintas
    - **Peak Hour**: Jam sibuk (11-14 dan 17-20)
    - **Weekend**: Hari weekend berdasarkan bulan tertentu
    """)

if __name__ == "__main__":
    predict_estimated_duration()
