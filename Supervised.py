import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

# Load data
data = pd.read_excel('Train Data.xlsx')

# Menambahkan fitur Peak Hour
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)

# Menambahkan fitur Weekend/Non-Weekend
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)

# Pilih fitur yang relevan untuk analisis
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Estimated Duration (min)', 'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

# Ambil data fitur relevan
X = data[features].dropna()  # Pastikan tidak ada nilai NaN

# Target variabel untuk prediksi: 'Delay (min)'
data['Delay (min)'] = data['Delivery Duration (min)'] - data['Estimated Duration (min)']
y = data['Delay (min)']  # Target: Delay (min)

# Pisahkan data menjadi train set dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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

# Fitting grid search pada data pelatihan
grid_search.fit(X_train, y_train)

# Menyimpan model terbaik yang ditemukan
best_rf = grid_search.best_estimator_

# Menyimpan model terlatih ke file pickle
with open('best_rf_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf, model_file)

print(f"Best parameters found: {grid_search.best_params_}")
