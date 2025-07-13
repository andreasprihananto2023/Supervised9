import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Memuat model Random Forest yang telah dilatih
try:
    with open('best_rf_model.pkl', 'rb') as model_file:
        model_info = pickle.load(model_file)
        
    # Ekstrak informasi model
    if isinstance(model_info, dict):
        best_rf = model_info['model']
        features = model_info['features']
        n_features = model_info['n_features']
        
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.write(f"Expected features: {n_features}")
        st.sidebar.write("Feature order:")
        for i, feature in enumerate(features, 1):
            st.sidebar.write(f"{i}. {feature}")
            
        if 'model_performance' in model_info:
            perf = model_info['model_performance']
            st.sidebar.write(f"Model R² Score: {perf['r2']:.3f}")
    else:
        # Fallback untuk model lama
        best_rf = model_info
        features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                   'Distance (km)', 'Topping Density', 'Traffic Level', 
                   'Is Peak Hour', 'Is Weekend']
        n_features = len(features)
        st.sidebar.warning("⚠️ Using fallback feature configuration")
        
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

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

    # Tombol untuk prediksi
    if st.button("Prediksi Estimasi Durasi", type="primary"):
        try:
            # Debug: cek jumlah fitur yang diharapkan model
            st.write("**Debug Info:**")
            st.write(f"Model expects {best_rf.n_features_in_} features")
            
            if hasattr(best_rf, 'feature_names_in_'):
                st.write(f"Expected features: {list(best_rf.feature_names_in_)}")
            
            # QUICK FIX: Check if model expects 9 features (including target variable)
            if best_rf.n_features_in_ == 9:
                st.warning("⚠️ Model was trained with target variable included. Using workaround...")
                # Add a dummy value for 'Estimated Duration (min)' at position 3
                # Order based on the expected features shown in debug
                input_data = np.array([[pizza_complexity, order_hour, restaurant_avg_time, 
                                        0,  # Dummy value for 'Estimated Duration (min)'
                                        distance, topping_density, traffic_level, 
                                        is_peak_hour, is_weekend]])
                st.write("⚠️ Added dummy value for 'Estimated Duration (min)' at position 4")
            else:
                # Normal case - 8 features without target variable
                input_data = np.array([[pizza_complexity, order_hour, restaurant_avg_time, 
                                        distance, topping_density, traffic_level, 
                                        is_peak_hour, is_weekend]])
            
            # Check the shape and print the features
            st.write(f"Input data shape: {input_data.shape}")
            st.write(f"Input features: {input_data[0]}")
            
            # Prediksi estimated duration menggunakan model
            predicted_duration = best_rf.predict(input_data)
            
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
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Debug info:")
            st.write(f"Input data shape: {input_data.shape}")
            st.write(f"Input data: {input_data}")
    
    # Informasi tambahan
    st.markdown("---")
    st.markdown("### 📊 Informasi Fitur")
    
    # Show different info based on model type
    if best_rf.n_features_in_ == 9:
        st.warning("⚠️ **Model Issue Detected**: This model was trained with the target variable included.")
        st.markdown("""
        **Recommended Action**: Retrain the model using the corrected training script.
        
        **Current Workaround**: The app adds a dummy value for 'Estimated Duration (min)' during prediction.
        """)
    
    st.markdown("""
    **Input features untuk prediksi**:
    1. **Pizza Complexity**: Tingkat kesulitan pembuatan pizza
    2. **Order Hour**: Jam pemesanan (mempengaruhi beban kerja)
    3. **Restaurant Avg Time**: Rata-rata waktu persiapan restoran
    4. **Distance**: Jarak pengiriman
    5. **Topping Density**: Kepadatan topping pada pizza
    6. **Traffic Level**: Kondisi lalu lintas
    7. **Peak Hour**: Jam sibuk (11-14 dan 17-20)
    8. **Weekend**: Hari weekend berdasarkan bulan tertentu
    """)
    
    # Recommendation box
    st.info("""
    💡 **Rekomendasi**: Untuk hasil yang optimal, retrain model menggunakan script yang telah diperbaiki 
    yang tidak menyertakan target variable dalam features.
    """)

if __name__ == "__main__":
    predict_estimated_duration()
