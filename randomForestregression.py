import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
file_path = "train_1.csv"  # Update this with the correct path if necessary
data = pd.read_csv(file_path)

# Prepare the inputs (weighted current frequency and the 10 frequency values) and outputs (r, g, b)
data['weighted_currFreq'] = data['currFreq'] * 1.5
X = data[['weighted_currFreq', 'freq1', 'freq2', 'freq3', 'freq4',
          'freq5', 'freq6', 'freq7', 'freq8', 'freq9', 'freq10']]
y = data[['r', 'g', 'b']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the values
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f'Mean Squared Error for R: {mse[0]:.2f}')
print(f'Mean Squared Error for G: {mse[1]:.2f}')
print(f'Mean Squared Error for B: {mse[2]:.2f}')

# If you want to see the predicted vs actual values
predictions = pd.DataFrame(y_pred, columns=['pred_r', 'pred_g', 'pred_b'])
actuals = y_test.reset_index(drop=True)
comparison = pd.concat([actuals, predictions], axis=1)
print(comparison)
