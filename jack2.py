import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
import keras_tuner as kt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

# 1. Dataset Collection
print("Fetching Dataset...")
ticker = "^NSEI"  
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Flatten MultiIndex columns introduced in recent yfinance updates
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# 2. Feature Expansion (The FRPS-Tech Framework)
print("Applying Feature Expansion...")
data['Pivot'] = (data['High'].squeeze() + data['Low'].squeeze() + data['Close'].squeeze()) / 3
data['R1'] = (2 * data['Pivot']) - data['Low'].squeeze()
data['S1'] = (2 * data['Pivot']) - data['High'].squeeze()

rolling_max = data['High'].squeeze().rolling(window=20).max()
rolling_min = data['Low'].squeeze().rolling(window=20).min()
diff = rolling_max - rolling_min
data['Fibo_236'] = rolling_max - diff * 0.236
data['Fibo_382'] = rolling_max - diff * 0.382
data['Fibo_618'] = rolling_max - diff * 0.618

data['SMA_20'] = data['Close'].squeeze().rolling(window=20).mean()
data['EMA_20'] = data['Close'].squeeze().ewm(span=20, adjust=False).mean()

data.dropna(inplace=True)
features = ['Close', 'Pivot', 'R1', 'S1', 'Fibo_236', 'Fibo_382', 'Fibo_618', 'SMA_20', 'EMA_20']
dataset = data[features].values

# 3. Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

def create_sequences(data_array, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data_array)):
        X.append(data_array[i-seq_length:i])
        y.append(data_array[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Model Architecture & Bayesian Optimization
def build_model(hp):
    model = Sequential()
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1)
    
    model.add(Bidirectional(LSTM(units=hp_units, return_sequences=False), 
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hp_dropout))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

print("Starting Bayesian Optimization...")
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=5, 
    directory='bilstm_tuning',
    project_name='frps_tech_optimization'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
best_model = tuner.get_best_models(num_models=1)[0]

# 5. Final Model Training
print("Training the optimized BiLSTM model...")
best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 6. Evaluation & Metrics
print("Evaluating Model...")
predictions = best_model.predict(X_test)

dummy_array = np.zeros((len(predictions), len(features)))
dummy_array[:, 0] = predictions.flatten()
inverse_predictions = scaler.inverse_transform(dummy_array)[:, 0]

dummy_array_y = np.zeros((len(y_test), len(features)))
dummy_array_y[:, 0] = y_test
inverse_y_test = scaler.inverse_transform(dummy_array_y)[:, 0]

mse = mean_squared_error(inverse_y_test, inverse_predictions)
mae = mean_absolute_error(inverse_y_test, inverse_predictions)
r2 = r2_score(inverse_y_test, inverse_predictions)

mape = np.mean(np.abs((inverse_y_test - inverse_predictions) / inverse_y_test)) * 100
model_accuracy = 100 - mape

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Model Accuracy (100 - MAPE): {model_accuracy:.2f}%")

# 7. Explainable AI (XAI) Integration using SHAP (Manual Plotting)
print("\nInitializing SHAP XAI Explainer...")

np.random.seed(42)
background_indices = np.random.choice(X_train.shape[0], 100, replace=False)
background = X_train[background_indices]

explainer = shap.GradientExplainer(best_model, background)

test_sample = X_test[:50]
print("Calculating SHAP values...")
shap_values = explainer.shap_values(test_sample)

if isinstance(shap_values, list):
    sv_array = shap_values[0] 
else:
    sv_array = shap_values 

mean_abs_shap = np.mean(np.abs(sv_array), axis=(0, 1))

print("Generating XAI Visualizations...")

# Ensure the SHAP array is strictly 1D to prevent nesting errors
mean_abs_shap = np.array(mean_abs_shap).flatten()

# Get the sorted indices
sorted_indices = np.argsort(mean_abs_shap)

# BULLETPROOF FIX: Force everything into native Python types
# int(i) strips the NumPy integer type, str() guarantees pure text labels
sorted_features = [str(features[int(i)]) for i in sorted_indices]
sorted_scores = [float(mean_abs_shap[int(i)]) for i in sorted_indices]

# Draw the custom Matplotlib chart
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_scores, color='royalblue', edgecolor='black')
plt.xlabel("Mean Absolute SHAP Value (Impact on Price Prediction)", fontsize=12)
plt.ylabel("Technical Features", fontsize=12)
plt.title("BiLSTM Feature Importance (Global Explanation)", fontsize=14, fontweight='bold')

# Add grid lines for easier reading
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()