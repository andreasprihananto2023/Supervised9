import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

print("Starting model training...")
print("=" * 50)

# Check if data file exists
if not os.path.exists('Train Data.xlsx'):
    print("❌ Error: 'Train Data.xlsx' not found!")
    print("Please make sure the training data file is in the same directory.")
    exit()

# Load data
try:
    data = pd.read_excel('Train Data.xlsx')
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

print("\nColumns in dataset:")
print(data.columns.tolist())
print(f"\nDataset shape: {data.shape}")

# Check if required columns exist
required_columns = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                   'Distance (km)', 'Topping Density', 'Traffic Level', 
                   'Order Month', 'Estimated Duration (min)']

missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"\n❌ Missing required columns: {missing_columns}")
    print("Available columns:", data.columns.tolist())
    exit()

# Menambahkan fitur Peak Hour
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)

# Menambahkan fitur Weekend/Non-Weekend
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)

# IMPORTANT: Features for training (WITHOUT target variable)
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

print(f"\n📊 Features to be used ({len(features)} features):")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

# Verify all features exist
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"\n❌ Missing features: {missing_features}")
    exit()

# Prepare feature matrix (X) and target vector (y)
X = data[features].copy()
y = data['Estimated Duration (min)'].copy()

print(f"\n📈 Data shapes:")
print(f"Feature matrix (X): {X.shape}")
print(f"Target vector (y): {y.shape}")

# Remove rows with NaN values
print(f"\n🧹 Cleaning data...")
print(f"Before removing NaN - X: {X.shape}, y: {y.shape}")
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]
print(f"After removing NaN - X: {X_clean.shape}, y: {y_clean.shape}")

if len(X_clean) == 0:
    print("❌ No valid data remaining after cleaning!")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)
print(f"\n📊 Data split:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Initialize Random Forest
rf = RandomForestRegressor(random_state=42)

# Simplified parameter grid for faster training
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print(f"\n🔍 Starting hyperparameter tuning...")
print("This may take a few minutes...")

# GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=3, 
    n_jobs=-1, 
    verbose=1, 
    scoring='neg_mean_absolute_error'
)

# Train the model
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"\n✅ Training completed!")
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate model
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Mean Absolute Error: {mae:.2f} minutes")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Feature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")

# Save model
model_info = {
    'model': best_rf,
    'features': features,
    'feature_names': features,
    'n_features': len(features),
    'model_performance': {
        'mae': mae,
        'mse': mse,
        'r2': r2
    },
    'best_params': grid_search.best_params_
}

try:
    with open('best_rf_model.pkl', 'wb') as model_file:
        pickle.dump(model_info, model_file)
    print(f"\n✅ Model saved as 'best_rf_model.pkl'")
except Exception as e:
    print(f"\n❌ Error saving model: {e}")
    exit()

# Test the saved model
print(f"\n🧪 Testing saved model...")
try:
    with open('best_rf_model.pkl', 'rb') as model_file:
        loaded_model_info = pickle.load(model_file)
    
    loaded_model = loaded_model_info['model']
    
    # Test prediction with dummy data
    dummy_data = np.array([[3, 14, 25, 5, 2, 3, 1, 0]])  # 8 features
    dummy_pred = loaded_model.predict(dummy_data)
    
    print(f"✅ Model test successful!")
    print(f"Dummy prediction: {dummy_pred[0]:.2f} minutes")
    print(f"Model expects {loaded_model.n_features_in_} features")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")

print(f"\n🎉 Training complete!")
print(f"You can now run your Streamlit app with: streamlit run app.py")
