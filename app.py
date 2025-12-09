import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Image size used during training
IMG_SIZE = (224, 224)

# 1. Load saved model (new Keras format)
model = keras.models.load_model("ai_waste_classification_model.keras")

# Classes must match your folder names: data/train/O and data/train/R
class_names = ['O', 'R']   # O = Organic, R = Recyclable

# Pretty labels for display
label_pretty = {
    'O': 'Organic waste',
    'R': 'Recyclable waste'
}

# Which bin to use for each class
bin_map = {
    'O': 'Wet / Organic bin',
    'R': 'Dry / Recyclables bin'
}

# 2. Helper function for prediction
def predict_image(img: Image.Image):
    # Resize to training size
    img = img.resize(IMG_SIZE)
    arr = keras.preprocessing.image.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # add batch dimension
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    return class_names[idx], float(preds[0][idx])

# 3. Streamlit UI
st.title("AI Waste Classification Demo")
st.write("Upload a photo of waste to see whether it is **organic (O)** or **recyclable (R)** and which bin to use.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify"):
        label, conf = predict_image(image)

        nice_label = label_pretty.get(label, label)
        st.subheader(f"Predicted class: **{nice_label}**")
        st.write(f"Confidence: {conf*100:.2f}%")

        st.write("Suggested bin:", bin_map.get(label, "Check with supervisor"))
