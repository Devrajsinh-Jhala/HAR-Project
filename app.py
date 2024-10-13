import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from flask_cors import CORS

features_to_remove = [
    'fBodyGyroMag-skewness()',
    'fBodyGyroMag-kurtosis()',
    'fBodyGyroMag-mean()',
    'fBodyGyroMag-std()',
    'fBodyGyroMag-max()',
    'fBodyGyroMag-entropy()',
    'tBodyAccMag-mean()-X',
    'tBodyAccMag-std()-X',
    'tBodyAccMag-max()-X',
    'tBodyAccMag-min()-X',
    'tBodyAccMag-mad()-X',
    'fBodyAccMag-skewness()-X',
    'fBodyAccMag-kurtosis()-X',
    'tBodyAccMag-energy()-X',
    'fBodyAccMag-mean()-X',
    'fBodyAccMag-std()-X',
    'fBodyAccMag-max()-X',
    'fBodyAccMag-entropy()-X',
    'tBodyAccMag-mean()-Y',
    'tBodyAccMag-std()-Y',
    'tBodyAccMag-max()-Y',
    'tBodyAccMag-min()-Y',
    'tBodyAccMag-mad()-Y',
    'fBodyAccMag-skewness()-Y',
    'fBodyAccMag-kurtosis()-Y',
    'tBodyAccMag-energy()-Y',
    'fBodyAccMag-mean()-Y',
    'fBodyAccMag-std()-Y',
    'fBodyAccMag-max()-Y',
    'fBodyAccMag-entropy()-Y',
    'tBodyAccMag-mean()-Z',
    'tBodyAccMag-std()-Z',
    'tBodyAccMag-max()-Z',
    'tBodyAccMag-min()-Z',
    'tBodyAccMag-mad()-Z',
    'fBodyAccMag-skewness()-Z',
    'fBodyAccMag-kurtosis()-Z',
    'tBodyAccMag-energy()-Z',
    'fBodyAccMag-mean()-Z',
    'fBodyAccMag-std()-Z',
    'fBodyAccMag-max()-Z',
    'fBodyAccMag-entropy()-Z'
]
feature_mapping = {
    'accel_x_mean': 'tBodyAcc-mean()-X',
    'accel_x_std': 'tBodyAcc-std()-X',
    'accel_x_max': 'tBodyAcc-max()-X',
    'accel_x_min': 'tBodyAcc-min()-X',
    'accel_x_mad': 'tBodyAcc-mad()-X',
    'accel_x_skewness': 'fBodyAcc-skewness()-X',
    'accel_x_kurtosis': 'fBodyAcc-kurtosis()-X',
    'accel_x_energy': 'tBodyAcc-energy()-X',
    'accel_x_freq_mean': 'fBodyAcc-mean()-X',
    'accel_x_freq_std': 'fBodyAcc-std()-X',
    'accel_x_freq_max': 'fBodyAcc-max()-X',
    'accel_x_freq_entropy': 'fBodyAcc-entropy()-X',
    
    'accel_y_mean': 'tBodyAcc-mean()-Y',
    'accel_y_std': 'tBodyAcc-std()-Y',
    'accel_y_max': 'tBodyAcc-max()-Y',
    'accel_y_min': 'tBodyAcc-min()-Y',
    'accel_y_mad': 'tBodyAcc-mad()-Y',
    'accel_y_skewness': 'fBodyAcc-skewness()-Y',
    'accel_y_kurtosis': 'fBodyAcc-kurtosis()-Y',
    'accel_y_energy': 'tBodyAcc-energy()-Y',
    'accel_y_freq_mean': 'fBodyAcc-mean()-Y',
    'accel_y_freq_std': 'fBodyAcc-std()-Y',
    'accel_y_freq_max': 'fBodyAcc-max()-Y',
    'accel_y_freq_entropy': 'fBodyAcc-entropy()-Y',

    'accel_z_mean': 'tBodyAcc-mean()-Z',
    'accel_z_std': 'tBodyAcc-std()-Z',
    'accel_z_max': 'tBodyAcc-max()-Z',
    'accel_z_min': 'tBodyAcc-min()-Z',
    'accel_z_mad': 'tBodyAcc-mad()-Z',
    'accel_z_skewness': 'fBodyAcc-skewness()-Z',
    'accel_z_kurtosis': 'fBodyAcc-kurtosis()-Z',
    'accel_z_energy': 'tBodyAcc-energy()-Z',
    'accel_z_freq_mean': 'fBodyAcc-mean()-Z',
    'accel_z_freq_std': 'fBodyAcc-std()-Z',
    'accel_z_freq_max': 'fBodyAcc-max()-Z',
    'accel_z_freq_entropy': 'fBodyAcc-entropy()-Z',

    'accel_magnitude_mean': 'tBodyAccMag-mean()',
    'accel_magnitude_std': 'tBodyAccMag-std()',
    'accel_magnitude_max': 'tBodyAccMag-max()',
    'accel_magnitude_min': 'tBodyAccMag-min()',
    'accel_magnitude_mad': 'tBodyAccMag-mad()',
    'accel_magnitude_skewness': 'fBodyAccMag-skewness()',
    'accel_magnitude_kurtosis': 'fBodyAccMag-kurtosis()',
    'accel_magnitude_energy': 'tBodyAccMag-energy()',
    'accel_magnitude_freq_mean': 'fBodyAccMag-mean()',
    'accel_magnitude_freq_std': 'fBodyAccMag-std()',
    'accel_magnitude_freq_max': 'fBodyAccMag-max()',
    'accel_magnitude_freq_entropy': 'fBodyAccMag-entropy()',

    'gyro_x_mean': 'tBodyGyro-mean()-X',
    'gyro_x_std': 'tBodyGyro-std()-X',
    'gyro_x_max': 'tBodyGyro-max()-X',
    'gyro_x_min': 'tBodyGyro-min()-X',
    'gyro_x_mad': 'tBodyGyro-mad()-X',
    'gyro_x_skewness': 'fBodyGyro-skewness()-X',
    'gyro_x_kurtosis': 'fBodyGyro-kurtosis()-X',
    'gyro_x_energy': 'tBodyGyro-energy()-X',
    'gyro_x_freq_mean': 'fBodyGyro-mean()-X',
    'gyro_x_freq_std': 'fBodyGyro-std()-X',
    'gyro_x_freq_max': 'fBodyGyro-max()-X',
    'gyro_x_freq_entropy': 'fBodyGyro-entropy()-X',

    'gyro_y_mean': 'tBodyGyro-mean()-Y',
    'gyro_y_std': 'tBodyGyro-std()-Y',
    'gyro_y_max': 'tBodyGyro-max()-Y',
    'gyro_y_min': 'tBodyGyro-min()-Y',
    'gyro_y_mad': 'tBodyGyro-mad()-Y',
    'gyro_y_skewness': 'fBodyGyro-skewness()-Y',
    'gyro_y_kurtosis': 'fBodyGyro-kurtosis()-Y',
    'gyro_y_energy': 'tBodyGyro-energy()-Y',
    'gyro_y_freq_mean': 'fBodyGyro-mean()-Y',
    'gyro_y_freq_std': 'fBodyGyro-std()-Y',
    'gyro_y_freq_max': 'fBodyGyro-max()-Y',
    'gyro_y_freq_entropy': 'fBodyGyro-entropy()-Y',

    'gyro_z_mean': 'tBodyGyro-mean()-Z',
    'gyro_z_std': 'tBodyGyro-std()-Z',
    'gyro_z_max': 'tBodyGyro-max()-Z',
    'gyro_z_min': 'tBodyGyro-min()-Z',
    'gyro_z_mad': 'tBodyGyro-mad()-Z',
    'gyro_z_skewness': 'fBodyGyro-skewness()-Z',
    'gyro_z_kurtosis': 'fBodyGyro-kurtosis()-Z',
    'gyro_z_energy': 'tBodyGyro-energy()-Z',
    'gyro_z_freq_mean': 'fBodyGyro-mean()-Z',
    'gyro_z_freq_std': 'fBodyGyro-std()-Z',
    'gyro_z_freq_max': 'fBodyGyro-max()-Z',
    'gyro_z_freq_entropy': 'fBodyGyro-entropy()-Z',

    'gyro_magnitude_mean': 'tBodyGyroMag-mean()',
    'gyro_magnitude_std': 'tBodyGyroMag-std()',
    'gyro_magnitude_max': 'tBodyGyroMag-max()',
    'gyro_magnitude_min': 'tBodyGyroMag-min()',
    'gyro_magnitude_mad': 'tBodyGyroMag-mad()',
    'gyro_magnitude_skewness': 'fBodyGyroMag-skewness()',
    'gyro_magnitude_kurtosis': 'fBodyGyroMag-kurtosis()',
    'gyro_magnitude_energy': 'tBodyGyroMag-energy()',
    'gyro_magnitude_freq_mean': 'fBodyGyroMag-mean()',
    'gyro_magnitude_freq_std': 'fBodyGyroMag-std()',
    'gyro_magnitude_freq_max': 'fBodyGyroMag-max()',
    'gyro_magnitude_freq_entropy': 'fBodyGyroMag-entropy()',

    'mag_x_mean': 'tBodyAccMag-mean()-X',
    'mag_x_std': 'tBodyAccMag-std()-X',
    'mag_x_max': 'tBodyAccMag-max()-X',
    'mag_x_min': 'tBodyAccMag-min()-X',
    'mag_x_mad': 'tBodyAccMag-mad()-X',
    'mag_x_skewness': 'fBodyAccMag-skewness()-X',
    'mag_x_kurtosis': 'fBodyAccMag-kurtosis()-X',
    'mag_x_energy': 'tBodyAccMag-energy()-X',
    'mag_x_freq_mean': 'fBodyAccMag-mean()-X',
    'mag_x_freq_std': 'fBodyAccMag-std()-X',
    'mag_x_freq_max': 'fBodyAccMag-max()-X',
    'mag_x_freq_entropy': 'fBodyAccMag-entropy()-X',

    'mag_y_mean': 'tBodyAccMag-mean()-Y',
    'mag_y_std': 'tBodyAccMag-std()-Y',
    'mag_y_max': 'tBodyAccMag-max()-Y',
    'mag_y_min': 'tBodyAccMag-min()-Y',
    'mag_y_mad': 'tBodyAccMag-mad()-Y',
    'mag_y_skewness': 'fBodyAccMag-skewness()-Y',
    'mag_y_kurtosis': 'fBodyAccMag-kurtosis()-Y',
    'mag_y_energy': 'tBodyAccMag-energy()-Y',
    'mag_y_freq_mean': 'fBodyAccMag-mean()-Y',
    'mag_y_freq_std': 'fBodyAccMag-std()-Y',
    'mag_y_freq_max': 'fBodyAccMag-max()-Y',
    'mag_y_freq_entropy': 'fBodyAccMag-entropy()-Y',

    'mag_z_mean': 'tBodyAccMag-mean()-Z',
    'mag_z_std': 'tBodyAccMag-std()-Z',
    'mag_z_max': 'tBodyAccMag-max()-Z',
    'mag_z_min': 'tBodyAccMag-min()-Z',
    'mag_z_mad': 'tBodyAccMag-mad()-Z',
    'mag_z_skewness': 'fBodyAccMag-skewness()-Z',
    'mag_z_kurtosis': 'fBodyAccMag-kurtosis()-Z',
    'mag_z_energy': 'tBodyAccMag-energy()-Z',
    'mag_z_freq_mean': 'fBodyAccMag-mean()-Z',
    'mag_z_freq_std': 'fBodyAccMag-std()-Z',
    'mag_z_freq_max': 'fBodyAccMag-max()-Z',
    'mag_z_freq_entropy': 'fBodyAccMag-entropy()-Z',

    'mag_magnitude_mean': 'tBodyAccMag-mean()',
    'mag_magnitude_std': 'tBodyAccMag-std()',
    'mag_magnitude_max': 'tBodyAccMag-max()',
    'mag_magnitude_min': 'tBodyAccMag-min()',
    'mag_magnitude_mad': 'tBodyAccMag-mad()',
    'mag_magnitude_skewness': 'fBodyAccMag-skewness()',
    'mag_magnitude_kurtosis': 'fBodyAccMag-kurtosis()',
    'mag_magnitude_energy': 'tBodyAccMag-energy()',
    'mag_magnitude_freq_mean': 'fBodyAccMag-mean()',
    'mag_magnitude_freq_std': 'fBodyAccMag-std()',
    'mag_magnitude_freq_max': 'fBodyAccMag-max()',
    'mag_magnitude_freq_entropy': 'fBodyAccMag-entropy()',
}






# Create FastAPI app
app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class SensorData(BaseModel):
    sensor_data: List[dict]

# Function to extract features from sensor data (similar to the one you had)
def extract_features(sensor_data):
    # Convert list of sensor data into a DataFrame
    df = pd.DataFrame(sensor_data)
    
    features = {}  # Store features as a dictionary initially
    
    def calculate_statistics(axis_data, axis_name):
        """Calculates time-domain statistics for each axis."""
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_mad'] = np.mean(np.abs(axis_data - np.mean(axis_data)))  # Mean Absolute Deviation
        features[f'{axis_name}_skewness'] = skew(axis_data)  # Skewness
        features[f'{axis_name}_kurtosis'] = kurtosis(axis_data)  # Kurtosis
        features[f'{axis_name}_energy'] = np.sum(axis_data**2) / len(axis_data)  # Signal energy

    def calculate_magnitude_features(x_data, y_data, z_data, axis_name):
        """Calculates magnitude features (derived from x, y, z)."""
        magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        calculate_statistics(magnitude, f'{axis_name}_magnitude')
        calculate_frequency_features(magnitude, f'{axis_name}_magnitude')
    
    def calculate_frequency_features(axis_data, axis_name):
        """Calculates frequency-domain features using Welch's method."""
        freqs, power = welch(axis_data, fs=50, nperseg=256)  # 50 Hz sampling rate
        features[f'{axis_name}_freq_mean'] = np.mean(power)
        features[f'{axis_name}_freq_std'] = np.std(power)
        features[f'{axis_name}_freq_max'] = np.max(power)
        features[f'{axis_name}_freq_entropy'] = -np.sum(power * np.log2(power + 1e-8))  # Spectral entropy
        
    # Split the data based on sensor type
    accel_data = df[df['sensor'] == 'accelerometer']
    gyro_data = df[df['sensor'] == 'gyroscope']
    mag_data = df[df['sensor'] == 'magnetometer']

    # Process accelerometer data (x, y, z)
    if not accel_data.empty:
        for axis in ['x', 'y', 'z']:
            calculate_statistics(accel_data[axis], f'accel_{axis}')
            calculate_frequency_features(accel_data[axis], f'accel_{axis}')
        calculate_magnitude_features(accel_data['x'], accel_data['y'], accel_data['z'], 'accel')

    # Process gyroscope data (x, y, z)
    if not gyro_data.empty:
        for axis in ['x', 'y', 'z']:
            calculate_statistics(gyro_data[axis], f'gyro_{axis}')
            calculate_frequency_features(gyro_data[axis], f'gyro_{axis}')
        calculate_magnitude_features(gyro_data['x'], gyro_data['y'], gyro_data['z'], 'gyro')

    # Process magnetometer data (x, y, z)
    if not mag_data.empty:
        for axis in ['x', 'y', 'z']:
            calculate_statistics(mag_data[axis], f'mag_{axis}')
            calculate_frequency_features(mag_data[axis], f'mag_{axis}')
        calculate_magnitude_features(mag_data['x'], mag_data['y'], mag_data['z'], 'mag')

    # Convert features dictionary to a DataFrame for easy reading
    features_df = pd.DataFrame([features])

    # Filter features to match your desired output
    columns_to_keep = list(feature_mapping.keys())

# Filter out only the columns to keep based on the keys
    features_df = features_df[columns_to_keep]

# Rename the columns from keys to their corresponding values
    features_df = features_df.rename(columns=feature_mapping)

    features_df = features_df.drop(columns=features_to_remove)
    return features_df

# Function to handle prediction request
def predict_activity(sensor_data):
    features_df = extract_features(sensor_data)
    

    # Load your pre-trained model
    model = joblib.load('random_forest_model.joblib')

    # Perform prediction
    prediction = model.predict(features_df)
    return {
        "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    }

# FastAPI route for predictions
@app.post("/predict")
async def predict(sensor_data: SensorData):
    sensor_data = sensor_data.sensor_data
    if not sensor_data:
        return {"error": "No sensor data provided"}
    print("Got Sensor data")

    # Get prediction
    prediction = predict_activity(sensor_data)
    print(prediction)
    print("Prediction to be sent: ",prediction)
    return {"prediction": prediction}
