import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.api.callbacks import ModelCheckpoint

# Specify the CSV file path
csv_file_path = 'features_raw.csv'

# Load the data
df = pd.read_csv(csv_file_path)

# Assuming the last column is the target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build a more complex neural network model with Conv1D
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Assuming a regression problem
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up model checkpointing
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[checkpoint])

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Calculate additional metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Print a few test inputs and their corresponding outputs
print("\nSample Test Inputs and Outputs:")
for i in range(5):  # Print the first 5 test samples
    print(f"Test Input {i+1}: {X_test[i].flatten()}")
    print(f"Predicted Output {i+1}: {y_pred[i]}")
    print(f"Actual Output {i+1}: {y_test[i]}")
    print()