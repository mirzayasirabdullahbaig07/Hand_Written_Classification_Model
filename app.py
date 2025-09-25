import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2

# ----------------------------
# Load MNIST model
# ----------------------------
MODEL_PATH = "mnist_model.keras"  # or "mnist_model.h5"
model = load_model(MODEL_PATH)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="centered"
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("✍️ About this Project")
st.sidebar.info(
    """
This project is a **handwritten digit classifier** trained on the **MNIST dataset**.
You can draw a digit (0-9) or upload an image and the model will predict it for you.
"""
)

st.sidebar.title("👤 About Me")
st.sidebar.markdown("""
**Mirza Yasir Abdullah Baig**  
- [LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/)  
- [GitHub](https://github.com/mirzayasirabdullahbaig07)  
- [Kaggle](https://www.kaggle.com/code/mirzayasirabdullah07)  
""")

# ----------------------------
# Main app
# ----------------------------
st.title("✍️ MNIST Handwritten Digit Classifier")
st.write("Draw a digit (0-9) below or upload an image, then click 'Predict'.")

# ----------------------------
# Drawing canvas
# ----------------------------
canvas_result = st_canvas(
    fill_color="#000000",       # black background
    stroke_width=15,
    stroke_color="#FFFFFF",     # white digit
    background_color="#000000",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

# ----------------------------
# Image uploader
# ----------------------------
# uploaded_file = st.file_uploader("Or upload a digit image (0-9)", type=["png", "jpg", "jpeg"])

# ----------------------------
# Helper function: preprocess image
# ----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # resize
    img_array = np.array(img)
    img_array = 255 - img_array  # invert colors
    img_array = (img_array > 50).astype(float)  # threshold
    img_array = img_array.reshape(1, 28, 28, 1)  # shape for model
    return img_array

# ----------------------------
# Predict button
# ----------------------------
if st.button("Predict"):
    img_array = None

    # Case 1: canvas drawing
    if canvas_result.image_data is not None:
        img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, :3]))
        img_array = preprocess_image(img)

    # Case 2: uploaded image
    elif uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)

    # Make prediction
    if img_array is not None:
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        st.success(f"Predicted Digit: {predicted_digit}")
    else:
        st.warning("Please draw a digit or upload an image!")
