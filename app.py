import streamlit as st
import src.ui.section_concept as section_concept
import src.ui.section_image as section_image
import src.ui.section_distribution as section_distribution

st.set_page_config(layout="wide", page_title="Visual Quantization Insight")

st.title("Visual Quantization Insight")
st.markdown("""
This application visualizes how **Quantization** works in Deep Learning.
Quantization maps continuous Floating Point numbers (high precision) to discrete Integers (low precision) to save memory and speed up computation.
""")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["1. Concept: Number Line", "2. Visual: Image Quantization", "3. Insight: Distribution"])

if options == "1. Concept: Number Line":
    section_concept.render()
elif options == "2. Visual: Image Quantization":
    section_image.render()
elif options == "3. Insight: Distribution":
    section_distribution.render()
