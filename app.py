"""
Photo Cartoonizer - Digital Image Processing Project
Pipeline:  Original -> Edge Mask -> Color Quantization -> Smoothing -> Cartoon
Run with:  streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st


# ---------------- STEP 1: Edge Detection ----------------
def get_edges(img, line_size=7, blur_value=7):
    """Find bold outlines, like the ink lines in a cartoon."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)          # colour -> grayscale
    gray_blur = cv2.medianBlur(gray, blur_value)           # remove noise, keep edges
    edges = cv2.adaptiveThreshold(                         # black lines on white
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        line_size, blur_value)
    return edges


# ---------------- STEP 2: Color Quantization ----------------
def quantize_colors(img, k=8):
    """Reduce the image to only k colours using K-Means clustering."""
    data = np.float32(img).reshape((-1, 3))                # every pixel = a 3D point (R,G,B)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)                            # the k "paint" colours
    result = centers[labels.flatten()]                     # replace each pixel by its cluster colour
    return result.reshape(img.shape)


# ---------------- STEP 3: Smoothing ----------------
def smooth(img, d=7):
    """Bilateral filter: smooths flat areas but keeps edges sharp."""
    return cv2.bilateralFilter(img, d=d, sigmaColor=200, sigmaSpace=200)


# ---------------- STEP 4: Combine ----------------
def cartoonize(img, k, line_size, blur_value, d):
    edges = get_edges(img, line_size, blur_value)
    colors = smooth(quantize_colors(img, k), d)
    cartoon = cv2.bitwise_and(colors, colors, mask=edges)  # draw black lines on top
    return edges, colors, cartoon


def resize_if_large(img, max_side=800):
    """Big photos make K-Means slow, so shrink them first."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ======================= USER INTERFACE =======================
st.set_page_config(page_title="Photo Cartoonizer", page_icon="🎨", layout="wide")
st.title("🎨 Photo Cartoonizer")
st.write("Upload a photo and watch it turn into a cartoon, step by step.")

# Sidebar sliders (odd values only, because OpenCV kernels need odd sizes)
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of colours (K)", 2, 16, 8)
line_size = st.sidebar.slider("Edge line size", 3, 15, 7, step=2)
blur_value = st.sidebar.slider("Edge blur", 3, 15, 7, step=2)
d = st.sidebar.slider("Smoothing strength", 3, 15, 7, step=2)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)       # OpenCV loads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # convert to RGB for display
    img = resize_if_large(img)

    with st.spinner("Cartoonizing..."):
        edges, colors, cartoon = cartoonize(img, k, line_size, blur_value, d)

    # Show every stage of the pipeline
    c1, c2, c3, c4 = st.columns(4)
    c1.image(img, caption="1. Original")
    c2.image(edges, caption="2. Edge mask")
    c3.image(colors, caption="3. Reduced colours")
    c4.image(cartoon, caption="4. Cartoon!")

    # Download button
    _, buf = cv2.imencode(".png", cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
    st.download_button("⬇️ Download cartoon", buf.tobytes(),
                       file_name="cartoon.png", mime="image/png")
else:
    st.info("👆 Upload a photo to get started.")
