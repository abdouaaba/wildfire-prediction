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

# Predict
def predict(model, image):
    prediction = model.predict(image)
    response = "Wildfire Risk Percentage: {:.2f}%".format(100 * prediction[0][1])
    return response

# Create the Streamlit app
def main():
    st.title("Wilfire Risk Prediction")

    # Allow user to upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    with st.sidebar:
        st.header("Input Coordinates")
        # Add input fields for longitude and latitude
        longitude = st.number_input("Longitude", value=0.0, step=0.0001, format="%.6f")
        latitude = st.number_input("Latitude", value=0.0, step=0.0001, format="%.6f")
        # Add a button to trigger the app
        button_clicked = st.button("Submit")

    # Wait for the button to be clicked
    if button_clicked:
        # load_dotenv()
        # Predict using coordinates
        api_key = st.secrets["api_key"]
        center = str(longitude) + ',' + str(latitude)
        rest = ',15,0/350x350?access_token=' + api_key + '&logo=false&attribution=false'
        url = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/' + center + rest

        # load & preprocess the image so the model can treat it properly
        response = requests.get(url)
        image = PIL.Image.open(BytesIO(response.content))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image.astype('float32')
        image = image / 255.0

        # Load and classify the image using the pre-trained model
        model, _ = load_data("first_model.hdf5", "dummy.jpg")
        result = predict(model, image)

        # Display the classification result
        st.write("Predicting...")
        st.write(result)

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
        result = predict(model, image)

        # Display the classification result
        st.write("Predicting...")
        st.write(result)

if __name__ == "__main__":
    main()
