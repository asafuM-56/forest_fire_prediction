import subprocess
import sys

# Force install scikit-learn if not found
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn  # Import again after installation

import gradio as gr
import pandas as pd
import pickle

# Load the pre-trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

def predict_forest_fire(temperature, humidity, wind_speed, rainfall, fuel_moisture, vegetation, slope, region, size, duration):
    # Creating input DataFrame for the model
    input_data = pd.DataFrame({
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'Wind Speed (km/h)': [wind_speed],
        'Rainfall (mm)': [rainfall],
        'Fuel Moisture (%)': [fuel_moisture],
        'Vegetation Type': [vegetation],
        'Slope (%)': [slope],
        'Region': [region],
        'Fire Size (hectares)': [size],
        'Fire Duration (hours)': [duration]
    })

    # One-hot encode the input data (ensure it matches the training data)
    input_encoded = pd.get_dummies(input_data)

    # Align columns with the training data (required columns)
    required_columns = model.feature_names_in_  # Get the feature columns from the model
    for col in required_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[required_columns]

    # Make the prediction
    prediction = model.predict(input_encoded)[0]

    # Reverse the label encoding (map the prediction back to the coffee type)
    forest_fire = label_encoder.inverse_transform([prediction])[0]

    return forest_fire

# Gradio Interface using components
interface = gr.Interface(
    fn=predict_forest_fire,
    inputs=[
        gr.Slider(minimum=0.0, maximum=50.0, step=0.5, label="Temperature (°C)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="Humidity (%)"),
        gr.Slider(minimum=0.0, maximum=50.0, step=0.5, label="Wind Speed (km/h)"),
        gr.Slider(minimum=0.0, maximum=50.0, step=0.5, label="Rainfall (mm)"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="Fuel Moisture (%)"),
        gr.Dropdown(['Grassland', 'Shrubland'], label="Vegetation Type"),
        gr.Slider(minimum=0.0, maximum=100.0, step=0.5, label="Slope (%)"),
        gr.Dropdown(['North', 'South', 'West'], label="Region"),
        gr.Slider(minimum=0.0, maximum=500.0, step=5.0, label="Fire Size (hectares)"),
        gr.Slider(minimum=0.0, maximum=150.0, step=1.0, label="Fire Duration (hours)")
    ],
    outputs=gr.Textbox(label="Forest Fire Prediction"),
    title="Forest Fire Prediction"
)

if __name__ == "__main__":
    interface.launch()
