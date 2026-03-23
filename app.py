# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# st.title("Weather Prediction")

# # Load models
# @st.cache_resource
# def load_models():
#     rf_model = joblib.load(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\rf_model.joblib")
#     scaler = joblib.load(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\scaler_rf.pkl")
#     cnn_model = tf.keras.models.load_model(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\weather_cnn_improved_model.h5")
#     lstm_model = tf.keras.models.load_model(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\rain_predict_lstm_model.h5")
#     return rf_model, scaler, cnn_model , lstm_model

# model, scaler, cnn_model, lstm_model = load_models()

# st.success("✓ Models loaded successfully")

# # Get class names from the CNN model (you may need to adjust this based on your training)
# class_names = ['cloudy', 'rain', 'shine', 'sunrise']  # Update with your actual classes

# st.markdown("---")
# st.subheader("Option 1: Predict from Weather Data")

# # Input fields
# col1, col2 = st.columns(2)

# with col1:
#     temperature_c = st.number_input('Temperature (C)', value=15.0)
#     apparent_temperature = st.number_input('Apparent Temperature (C)', value=14.0)
#     humidity = st.number_input('Humidity', min_value=0.0, max_value=1.0, value=0.5)
#     wind_speed = st.number_input('Wind Speed (km/h)', value=10.0)

# with col2:
#     wind_bearing = st.number_input('Wind Bearing (degrees)', value=180.0)
#     visibility = st.number_input('Visibility (km)', value=10.0)
#     loud_cover = st.number_input('Loud Cover', min_value=0.0, max_value=1.0, value=0.5)
#     pressure = st.number_input('Pressure (millibars)', value=1015.0)

# # Predict from data
# if st.button('Predict from Data', use_container_width=True):
#     input_data = [[temperature_c, apparent_temperature, humidity, wind_speed, 
#                    wind_bearing, visibility, loud_cover, pressure]]
    
#     input_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_scaled)[0]
#     prediction_proba = model.predict_proba(input_scaled)[0]
    
#     if prediction == 1:
#         st.success(f"🌧️ RAIN - Probability: {prediction_proba[1]*100:.1f}%")
#     else:
#         st.info(f"☀️ NO RAIN - Probability: {prediction_proba[0]*100:.1f}%")

# st.markdown("---")
# st.subheader("Option 2: Predict from Image")

# # Image upload
# uploaded_file = st.file_uploader("Upload a weather image", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_container_width=True)
    
#     # Predict button for image
#     if st.button('Predict from Image', use_container_width=True):
#         with st.spinner('Analyzing image...'):
#             # Preprocess image
#             img = image.convert('RGB')
#             img = img.resize((150, 150))  # Match training size
#             img_array = np.array(img)
#             img_array = img_array / 255.0  # Normalize
#             img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
#             # Make prediction
#             predictions = cnn_model.predict(img_array, verbose=0)
#             predicted_class_idx = np.argmax(predictions[0])
#             predicted_class = class_names[predicted_class_idx]
#             confidence = predictions[0][predicted_class_idx] * 100
            
#             # Display result
#             st.success(f"🌤️ Predicted Weather: **{predicted_class.upper()}**")
#             st.info(f"Confidence: {confidence:.1f}%")
            
#             # Show all probabilities
#             with st.expander("View all predictions"):
#                 for i, class_name in enumerate(class_names):
#                     st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")


import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.title("Weather Prediction")

# Load models
@st.cache_resource
def load_models():
    rf_model = joblib.load(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\rf_model.joblib")
    scaler = joblib.load(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\scaler_rf.pkl")
    cnn_model = tf.keras.models.load_model(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\weather_cnn_improved_model.h5")
    lstm_model = tf.keras.models.load_model(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\rain_predict_lstm_model.h5")
    
    # Load tokenizer (you need to save this during training)
    try:
        with open(r"C:\Users\vaibh\OneDrive\Desktop\latest_projects\weather_forecasting\tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)
    except:
        tokenizer = None
        
    return rf_model, scaler, cnn_model, lstm_model, tokenizer

model, scaler, cnn_model, lstm_model, tokenizer = load_models()

st.success("✓ Models loaded successfully")

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

# Helper function for LSTM
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.markdown("---")
st.subheader("Option 1: Predict from Weather Data")

col1, col2 = st.columns(2)

with col1:
    temperature_c = st.number_input('Temperature (C)', value=15.0)
    apparent_temperature = st.number_input('Apparent Temperature (C)', value=14.0)
    humidity = st.number_input('Humidity', min_value=0.0, max_value=1.0, value=0.5)
    wind_speed = st.number_input('Wind Speed (km/h)', value=10.0)

with col2:
    wind_bearing = st.number_input('Wind Bearing (degrees)', value=180.0)
    visibility = st.number_input('Visibility (km)', value=10.0)
    loud_cover = st.number_input('Loud Cover', min_value=0.0, max_value=1.0, value=0.5)
    pressure = st.number_input('Pressure (millibars)', value=1015.0)

if st.button('Predict from Data', use_container_width=True):
    input_data = [[temperature_c, apparent_temperature, humidity, wind_speed, 
                   wind_bearing, visibility, loud_cover, pressure]]
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    if prediction == 1:
        st.success(f"🌧️ RAIN - Probability: {prediction_proba[1]*100:.1f}%")
    else:
        st.info(f"☀️ NO RAIN - Probability: {prediction_proba[0]*100:.1f}%")

st.markdown("---")
st.subheader("Option 2: Predict from Image")

uploaded_file = st.file_uploader("Upload a weather image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict from Image', use_container_width=True):
        with st.spinner('Analyzing image...'):
            img = image.convert('RGB')
            img = img.resize((150, 150))
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = cnn_model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            st.success(f"🌤️ Predicted Weather: **{predicted_class.upper()}**")
            st.info(f"Confidence: {confidence:.1f}%")
            
            with st.expander("View all predictions"):
                for i, class_name in enumerate(class_names):
                    st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")

st.markdown("---")
st.subheader("Option 3: Predict from Text Description")

weather_text = st.text_area(
    "Enter weather description",
    placeholder="E.g., Partly cloudy throughout the day.",
    height=100
)

if st.button('Predict from Text', use_container_width=True):
    if weather_text and tokenizer is not None:
        with st.spinner('Analyzing text...'):
            # Clean and preprocess text
            cleaned = clean_text(weather_text)
            
            # Tokenize and pad
            max_len = 50
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            
            # Predict
            pred = lstm_model.predict(padded, verbose=0)[0][0]
            
            if pred > 0.5:
                st.success(f"🌧️ RAIN - Probability: {pred*100:.1f}%")
            else:
                st.info(f"☀️ NO RAIN - Probability: {(1-pred)*100:.1f}%")
    elif not tokenizer:
        st.error("Tokenizer not found. Please save tokenizer during training.")
    else:
        st.warning("Please enter a weather description.")