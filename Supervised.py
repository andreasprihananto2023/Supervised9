import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load data
data = pd.read_excel('Train Data.xlsx')

print("Columns in dataset:")
print(data.columns.tolist())
print(f"\nDataset shape: {data.shape}")

# Menambahkan fitur Peak Hour
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)

# Menambahkan fitur Weekend/Non-Weekend
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)

# PENTING: Definisi fitur yang akan digunakan (TANPA Estimated Duration)
# Removed 'Estimated Duration (min)' from features - this is our target variable!
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

print(f"\nFeatures to be used ({len(features)} features):")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

# Cek apakah semua fitur ada dalam dataset
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"\nMissing features: {missing_features}")
    exit()

# Ambil data fitur relevan (INPUT FEATURES ONLY)
X = data[features].copy()
print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature matrix columns: {X.columns.tolist()}")

# Target variabel untuk prediksi: 'Estimated Duration (min)'
y = data['Estimated Duration (min)'].copy()
print(f"Target variable shape: {y.shape}")

# Hapus baris yang memiliki NaN
print(f"\nBefore removing NaN - X shape: {X.shape}, y shape: {y.shape}")
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]
print(f"After removing NaN - X shape: {X.shape}, y shape: {y.shape}")

# Pisahkan data menjadi train set dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Inisialisasi dan latih model Random Forest
rf = RandomForestRegressor(random_state=0)

# Parameter yang akan dicari menggunakan GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],            # Jumlah pohon dalam hutan
    'max_depth': [None, 10, 20, 30],             # Kedalaman maksimum pohon
    'min_samples_split': [2, 5, 10],             # Minimal sampel untuk membagi node
    'min_samples_leaf': [1, 2, 4],               # Minimal sampel untuk menjadi daun
    'max_features': ['auto', 'sqrt', 'log2']    # Fitur maksimal yang dipertimbangkan
}

# Inisialisasi GridSearchCV dengan cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

print("\nStarting GridSearchCV...")
# Fitting grid search pada data pelatihan
grid_search.fit(X_train, y_train)

# Menyimpan model terbaik yang ditemukan
best_rf = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Model trained with {best_rf.n_features_in_} features")

# Evaluasi model pada test set
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Menyimpan model terlatih dan informasi fitur ke file pickle
model_info = {
    'model': best_rf,
    'features': features,
    'feature_names': features,
    'n_features': len(features),
    'model_performance': {
        'mae': mae,
        'mse': mse,
        'r2': r2
    }
}

with open('best_rf_model.pkl', 'wb') as model_file:
    pickle.dump(model_info, model_file)

print(f"\nModel saved successfully!")
print(f"Model expects {len(features)} features in this order:")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

# Test prediksi dengan data dummy
print("\nTesting model with dummy data...")
dummy_data = np.array([[3, 14, 25, 5, 2, 3, 1, 0]])  # 8 features
try:
    dummy_pred = best_rf.predict(dummy_data)
    print(f"Dummy prediction successful: {dummy_pred[0]:.2f} minutes")
except Exception as e:
    print(f"Dummy prediction failed: {e}")

print(f"\nVerification:")
print(f"Features used for training: {len(features)}")
print(f"Dummy data features: {dummy_data.shape[1]}")
print(f"Match: {len(features) == dummy_data.shape[1]}")
