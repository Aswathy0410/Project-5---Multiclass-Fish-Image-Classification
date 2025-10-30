import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import json

# Load the trained model
try:
    model = load_model('fish_classifier_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load class indices
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
except Exception as e:
    st.error(f"Error loading class indices: {e}")
    st.stop()

# Title of the app
st.title('Fish Image Classifier')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.write("Classifying...")

        # Preprocess the image
        img = image.load_img(uploaded_file, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence_score = np.max(prediction) * 100

        # Display results
        st.write(f'Predicted Fish Category: {predicted_class}')
        st.write(f'Confidence Score: {confidence_score:.2f}%')
    except Exception as e:
        st.error(f"Error during prediction: {e}")