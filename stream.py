import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\Pabitra\\Desktop\\potato project\\saved_models\\1\\model.keras")


# Define class labels
class_names = ["Healthy", "Early Blight", "Late Blight"]

# Streamlit app
st.title("Potato Leaf Disease Classification")

st.write("Upload an image of a potato leaf to classify it as Healthy, Early Blight, or Late Blight.")

# Prediction function
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Predict and display the result
    predicted_class, confidence = predict(model, image_data)
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
