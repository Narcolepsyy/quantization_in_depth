import streamlit as st
import torch
import matplotlib.pyplot as plt
import src.core.quantization as qu

def render():
    st.header("3. Impact on Data Distribution")
    st.markdown("See how quantization 'bins' data points. Notice how outliers (very large or small values) can stretch the scale, causing loss of precision for small values.")
    
    col1, col2 = st.columns(2)
    with col1:
        dist_type = st.selectbox("Distribution Type", ["Normal (Gaussian)", "Uniform", "With Outliers"])
        num_samples = 10000
        
        if dist_type == "Normal (Gaussian)":
            data = torch.randn(num_samples)
        elif dist_type == "Uniform":
            data = torch.rand(num_samples) * 10 - 5
        elif dist_type == "With Outliers":
            data = torch.cat([torch.randn(num_samples), torch.tensor([10.0, -10.0, 12.0])])

    with col2:
         q_dtype_name = st.selectbox("Quantization Type", ["int8", "int4 (Simulated)"])
         
    if q_dtype_name == "int8":
        dtype = torch.int8
        q_bits = 8
    else:
        # Simulating int4 with same logic but smaller range
        dtype = torch.int8 
        q_bits = 4

    # Get Params
    # For simulation of custom bits we calculate manually
    if q_bits == 8:
        scale, zero_point = qu.get_q_scale_and_zero_point(data, dtype=torch.int8)
        q_min, q_max = -128, 127
    else:
        # Custom logic for simulated int4
        q_min, q_max = -8, 7
        min_val, max_val = data.min().item(), data.max().item()
        scale = (max_val - min_val) / (q_max - q_min)
        if scale == 0: scale = 1.0
        zero_point = q_min - round(min_val / scale)
        zero_point = max(q_min, min(q_max, zero_point))

    # Apply Logic
    if q_bits == 8:
         q_data = qu.linear_q_with_scale_and_zero_point(data, scale, zero_point, torch.int8)
         deq_data = qu.linear_dequantization(q_data, scale, zero_point)
    else:
         # Manual quantization for simulated bits
         scaled = data / scale + zero_point
         rounded = torch.round(scaled).clamp(q_min, q_max)
         deq_data = scale * (rounded - zero_point)
    
    st.subheader("Histograms")
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Original Data
    ax[0].hist(data.numpy(), bins=100, color='blue', alpha=0.7, label='Original')
    ax[0].set_title("Original Distribution")
    ax[0].legend()
    
    # Dequantized Data
    ax[1].hist(deq_data.numpy(), bins=100, color='red', alpha=0.7, label='Dequantized (Quantized)')
    ax[1].set_title(f"After {q_bits}-bit Quantization")
    ax[1].legend()
    
    st.pyplot(fig)
    
    st.markdown(f"**Stats:**")
    st.write(f"Scale: {scale:.4f}, Zero Point: {zero_point}")
    mse = (data - deq_data).square().mean().item()
    st.write(f"Mean Squared Error: {mse:.6f}")
