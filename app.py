import streamlit as st
import numpy as np
import os
import requests
import PIL
from io import BytesIO
from dotenv import load_dotenv


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# Load the pre-trained model and image
def load_data(model_path, image_path):
    model = load_model(model_path)
    image = load_img(image_path, target_size=(350, 350))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image /= 255.0
    return model, image

# Classify the image
def classify_image(model, image):
    prediction = model.predict(image)
    if prediction[0][0] > prediction[0][1]:
        return "No risk of wildfire"
    else:
        return "Risk of wildfire"

# Create the Streamlit app
def main():
    st.title("Image Classification")

    # Allow user to upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = load_img(uploaded_file, target_size=(350, 350))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image.astype('float32')
        image /= 255.0

        # Load and classify the image using the pre-trained model
        model, _ = load_data("first_model.hdf5", "dummy.jpg")
        result = classify_image(model, image)

        # Display the classification result
        st.write("Classifying...")
        st.write(result)

if __name__ == "__main__":
    main()
