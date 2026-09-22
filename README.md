#  Photo Cartoonizer

A Digital Image Processing project that turns any photo into a cartoon using classic computer vision techniques with **OpenCV** and a simple **Streamlit** web interface.

The app shows every stage of the pipeline side by side, so you can see exactly how the cartoon effect is built.

##  Features
- Upload any JPG or PNG photo
- See each processing step: original → edge mask → reduced colours → cartoon
- Adjust settings live with sliders
- Download the final cartoon image

##  How It Works

| Step | Technique | Purpose |
|------|-----------|---------|
| 1. Edge Detection | Grayscale + Median Blur + Adaptive Thresholding | Creates bold black outlines |
| 2. Color Quantization | K-Means Clustering | Reduces the photo to K flat colours |
| 3. Smoothing | Bilateral Filter | Smooths colour areas while keeping edges sharp |
| 4. Combining | Bitwise AND with edge mask | Draws the outlines over the colours |

##  How to Run

```bash
git clone https://github.com/<your-username>/photo-cartoonizer.git
cd photo-cartoonizer
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Settings

| Slider | Effect |
|--------|--------|
| Number of colours (K) | Fewer = more cartoonish, more = more realistic |
| Edge line size | Controls thickness and detail of outlines |
| Edge blur | Higher = fewer, cleaner lines |
| Smoothing strength | Higher = flatter, smoother colours |

## Tech Stack
- Python 3.9+
- OpenCV
- NumPy
- Streamlit
