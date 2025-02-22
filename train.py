import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from keras.api.utils import to_categorical
from keras.api.models import Model
from keras.api.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, AveragePooling2D, SeparableConv2D, Flatten, Dense, Dropout

# Define the paths to the dataset folders
dataset_path = 'dataset'
ads_path = os.path.join(dataset_path, 'ads')
songs_path = os.path.join(dataset_path, 'songs')

# Function to load data from CSV files and create labels based on file names
def load_data_from_folder(folder_path):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, header=None)
            data.append(df.values)
            # Create label based on file name
            if 'Funny' in file_name:
                label = 0
            elif 'Scary' in file_name:
                label = 1
            elif 'Sad' in file_name:
                label = 2
            elif 'Enjoyed' in file_name:
                label = 3
            elif 'Relaxed' in file_name:
                label = 4
            labels.extend([label] * df.shape[0])
    data = np.concatenate(data, axis=0)
    labels = np.array(labels)
    return data, labels

# Load data from ads and songs folders
ads_data, ads_labels = load_data_from_folder(ads_path)
songs_data, songs_labels = load_data_from_folder(songs_path)

print(ads_data.shape, ads_labels.shape)
print(songs_data.shape, songs_labels.shape)

# Function to preprocess and train a model
def preprocess_and_train(X, y, model_name):
    # Check for NaN or infinite values in data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("NaN or infinite values found in data. Handling them...")
        X = np.nan_to_num(X)  # Replace NaN with 0 and infinite with large finite numbers

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data across time samples (flatten channels, then reshape back)
    scaler = StandardScaler()
    n_trials, n_timepoints = X_train.shape
    X_train_reshaped = X_train.reshape(n_trials, -1)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_trials, n_timepoints)
    n_trials_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(n_trials_test, -1)).reshape(n_trials_test, n_timepoints)

    # Reshape data for EEGNet (assuming 2 channels)
    n_channels = 2
    X_train_final = X_train_scaled.reshape(-1, n_timepoints // n_channels, n_channels, 1)
    X_test_final = X_test_scaled.reshape(-1, n_timepoints // n_channels, n_channels, 1)

    # Convert labels to categorical
    y_train_categorical = to_categorical(y_train, num_classes=5)
    y_test_categorical = to_categorical(y_test, num_classes=5)

    # Define EEGNet architecture
    def EEGNet(nb_classes, Chans=2, Samples=n_timepoints // n_channels, dropoutRate=0.5):
        input1 = Input(shape=(Samples, Chans, 1))
        
        # First block: Temporal Convolution
        block1 = Conv2D(16, (64, 1), padding='same', activation='elu')(input1)
        block1 = BatchNormalization()(block1)
        
        # Second block: Depthwise Convolution for spatial filtering
        block2 = DepthwiseConv2D((1, Chans), depth_multiplier=2, padding='valid', activation='elu')(block1)
        block2 = BatchNormalization()(block2)
        block2 = AveragePooling2D((2, 1))(block2)  # Adjusted pooling size
        block2 = Dropout(dropoutRate)(block2)
        
        # Third block: Separable Convolution
        block3 = SeparableConv2D(16, (16, 1), padding='same', activation='elu')(block2)
        block3 = BatchNormalization()(block3)
        block3 = AveragePooling2D((2, 1))(block3)  # Adjusted pooling size
        block3 = Dropout(dropoutRate)(block3)
        
        flatten = Flatten()(block3)
        dense = Dense(nb_classes, activation='softmax')(flatten)
        
        model = Model(inputs=input1, outputs=dense)
        return model

    # Instantiate EEGNet for multi-class classification (nb_classes=5)
    model = EEGNet(nb_classes=5, Chans=n_channels, Samples=n_timepoints // n_channels, dropoutRate=0.5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    checkpoint = ModelCheckpoint(f'best_{model_name}_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # Train the model
    model.fit(X_train_final, y_train_categorical, epochs=20, batch_size=16, validation_split=0.2,
              callbacks=[checkpoint, early_stop])

    # Load the best model weights
    model.load_weights(f'best_{model_name}_model.keras')

    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test_final, y_test_categorical)
    print(f"Test Loss ({model_name}): {loss}, Test Accuracy ({model_name}): {accuracy}")

    # Print a few test inputs and their corresponding outputs
    print(f"\nSample Test Inputs and Outputs ({model_name}):")
    for i in range(5):  # Print the first 5 test samples
        print(f"Test Input {i+1}: {X_test_final[i].flatten()}")
        print(f"Predicted Output {i+1}: {model.predict(X_test_final[i:i+1])[0]}")
        print(f"Actual Output {i+1}: {y_test[i]}")
        print()

# Preprocess and train models for ads and songs data separately
preprocess_and_train(ads_data, ads_labels, 'ads')
preprocess_and_train(songs_data, songs_labels, 'songs')