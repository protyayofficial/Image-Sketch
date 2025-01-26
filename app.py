from PIL import Image
import numpy as np
import cv2
import streamlit as st

class PencilSketch:
    """Apply a high-quality pencil sketch effect to an image."""

    def __init__(self, blur_sigma: int = 5, ksize: tuple = (21, 21), sharpen_value: int = 1) -> None:
        self.blur_sigma = blur_sigma
        self.ksize = ksize
        self.sharpen_value = sharpen_value

    def dodge(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        """Apply the dodge blend mode between two images."""
        result = front * 255.0 / (255.0 - back) 
        result[result > 255] = 255
        result[back == 255] = 255
        return result.astype('uint8')

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening using a kernel filter."""
        if self.sharpen_value is not None and isinstance(self.sharpen_value, int):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        return image

    def apply_sketch(self, frame: np.ndarray) -> np.ndarray:
        """Apply the full pencil sketch effect to the image."""
        
        # Step 1: Convert image to grayscale with contrast enhancement
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)  # Enhance contrast
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel grayscale

        # Step 2: Invert the grayscale image
        inverted_img = 255 - grayscale

        # Step 3: Apply Gaussian Blur to the inverted image
        blur_img = cv2.GaussianBlur(inverted_img, self.ksize, self.blur_sigma)

        # Step 4: Dodge the blurred image with the grayscale
        final_img = self.dodge(grayscale, blur_img)

        # Step 5: Apply sharpening to the final image
        sharpened_image = self.sharpen(final_img)

        return sharpened_image

# Streamlit UI to upload and display images
st.title("Pencil Sketch Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Create the PencilSketch instance with default parameters
    pencil_sketch = PencilSketch(blur_sigma=5, ksize=(21, 21), sharpen_value=2)
    
    # Apply pencil sketch effect
    sketch_image = pencil_sketch.apply_sketch(img_array)
    
    # Convert to PIL Image for display
    final_sketch = Image.fromarray(sketch_image)
    
    # Display the original and pencil sketch
    st.image(image, caption="Original Image", use_container_width=True)
    st.image(final_sketch, caption="Pencil Sketch Image", use_container_width=True)
