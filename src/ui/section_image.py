import streamlit as st
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import src.core.quantization as qu

def render():
    st.header("2. Visualizing Quantization on Images")
    st.markdown("Images are just 3D tensors (Height, Width, Color Channels). Quantizing them reduces the number of colors available, which can create 'banding' artifacts.")

    # Default image
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    else:
        st.info("Using default image. You can upload your own above.")
        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
             st.error("Could not load default image. Please upload one.")
             image = None

    if image:
        # Resize for performance in demo
        image = image.resize((512, 512))
        st.image(image, caption='Original Image', use_container_width=True)

        # Convert to tensor
        img_tensor = torch.tensor(np.array(image)).float()
        
        st.subheader("Quantization Settings")
        col1, col2 = st.columns(2)
        with col1:
             num_bits = st.slider("Bit-width (lower = more compression)", 1, 8, 4)
             
        # Quantize
        dequantized_tensor, scale, zero_point = qu.quantize_image(img_tensor, num_bits)
        
        # Convert back to uint8 for display
        dequantized_img_np = dequantized_tensor.numpy().clip(0, 255).astype(np.uint8)
        dequantized_image = Image.fromarray(dequantized_img_np)
        
        st.subheader("Results")
        c1, c2 = st.columns(2)
        
        with c1:
            st.image(dequantized_image, caption=f'Quantized Image ({num_bits}-bit)', use_container_width=True)
            
        with c2:
            # Error Map
            error_tensor = (img_tensor - dequantized_tensor).abs()
            # Normalize error for better visualization (0 to 255)
            error_img_np = (error_tensor.mean(dim=2).numpy())
            
            # Simple grayscale error map
            # Scale error for visibility: mult by 5 to exaggerate small errors
            error_display = (error_img_np * 5).clip(0, 255).astype(np.uint8)
            error_image_pil = Image.fromarray(error_display, mode='L')
            
            st.image(error_image_pil, caption='Error Map (Darker = Less Error)', use_container_width=True)
            
        st.metric("Scale used", f"{scale:.4f}")
        st.metric("Zero Point used", f"{zero_point}")
