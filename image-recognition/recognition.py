import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Function to load and preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Streamlit app
def main():
    st.title("Image Recognition App")
    st.write("Upload an image for recognition:")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        image = preprocess_image(uploaded_image)
        preds = model.predict(image)
        label = decode_predictions(preds, top=1)[0][0]
        st.write(f"Prediction: {label[1]}, Probability: {label[2]}")

if __name__ == "__main__":
    main()
