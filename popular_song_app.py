import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os

# Load trained model
model_url = 'https://drive.google.com/uc?export=download&id=1NxDS9c3ElqVFORdkG0gfAPRLnE-ty2OE'
model_path = "song_hit_model.pkl"


# Set feature names
features = ['danceability', 'energy', 'loudness', 'instrumentalness', 'tempo']

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(model_url, model_path)
rfc_model = joblib.load(model_path)
# App title and instructions
st.title("Hit Song Predictor")
st.write("Use the sliders to simulate a song and see if it's likely to be a hit.")

# User inputs via sliders
user_input = {
    'danceability': st.slider('Danceability (0-1)', 0.0, 1.0, 0.63),
    'energy': st.slider('Energy (0-1)', 0.0, 1.0, 0.64),
    'loudness': st.slider('Loudness (dB)', -60.0, 0.0, -6.9),
    'instrumentalness': st.slider('Instrumentalness (0-1)', 0.0, 1.0, 0.0),
    'tempo': st.slider('Tempo', 60, 200, 80)   
}

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Get probability of being a hit (class 1)
probability = rfc_model.predict_proba(input_df)[0][1]

# Set custom threshold
threshold = 0.06

# Display prediction
if probability > threshold:
    st.success(f"The song is likely to be a **HIT**!")
    
else:
    st.warning(f"The song might **not** be a hit.")
    