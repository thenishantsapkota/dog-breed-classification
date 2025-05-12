import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #3366ff;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
    }
    .result-container {
        background-color: #f0f5ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .confidence-meter {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .upload-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🐶 Dog Breed Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload a photo of a dog to identify its breed</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/20250512-19221747077723-dog-breed-classification.h5"
    )
    df = pd.read_csv("raw/labels.csv")
    class_names = sorted(df["breed"].drop_duplicates().values.tolist())
    return model, class_names


model, class_names = load_model()


def preprocess(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


with col1:
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("### Upload a dog image")
    uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "JPG"])
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        classify_btn = st.button("🔍 Identify Breed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### Popular Dog Breeds")
    st.markdown("""
    - Labrador Retriever
    - German Shepherd
    - Golden Retriever
    - Bulldog
    - Beagle
    - Poodle
    - Siberian Husky
    """)

with col2:
    if uploaded and classify_btn:
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        with st.spinner("🔍 Analyzing the image..."):
            input_tensor = preprocess(image)
            preds = model.predict(input_tensor)[0]
            
            top_indices = preds.argsort()[-3:][::-1]
            top_breeds = [class_names[i] for i in top_indices]
            top_confidences = [preds[i] * 100 for i in top_indices]
            
            st.markdown(f"### 🏆 Predicted Breed: **{top_breeds[0]}**")
            st.markdown(f"<div class='confidence-meter'>Confidence: {top_confidences[0]:.1f}%</div>", unsafe_allow_html=True)
            st.progress(float(top_confidences[0]/100))
            
            st.markdown("### Other possibilities:")
            for i in range(1, 3):
                st.markdown(f"- **{top_breeds[i]}** ({top_confidences[i]:.1f}% confidence)")
        
        st.markdown("""
        ### About this breed
        This prediction is based on visual patterns the AI has learned. For a definitive breed identification, 
        please consult with a veterinarian or dog breed expert.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif uploaded:
        st.info("👆 Click the 'Identify Breed' button to analyze the image")
    else:
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.markdown("""
        ### How to use this app:
        
        1. Upload a clear photo of a dog
        2. Click the "Identify Breed" button
        3. View the predicted breed and confidence score
        
        #### Tips for best results:
        - Use well-lit, clear images
        - Try to capture the full dog in the frame
        - Front-facing photos work best
        - Avoid blurry or dark images
        """)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("#### Made with ❤️ using TensorFlow and Streamlit by Nishant Sapkota")
st.markdown("© 2025 Dog Breed Classifier | For educational purposes only")