from PIL import Image
import cv2
import numpy as np
import streamlit as st
from PIL import Image

def sketch_image(image):
    # Convert PIL Image to cv2 format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    invert = cv2.bitwise_not(gray)
    
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    
    invertedBlur = cv2.bitwise_not(blur)
    
    sketch = cv2.divide(gray, invertedBlur, scale=256.0)
    
    # Convert back to PIL Image
    return Image.fromarray(sketch)

st.title("Image Sketch Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Generating sketch...")
    sketch = sketch_image(image)
    st.image(sketch, caption='Sketch Image', use_container_width=True)