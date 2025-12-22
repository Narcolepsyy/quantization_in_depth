import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import src.core.quantization as qu

def render():
    st.header("1. The Concept: Mapping Floats to Ints")
    st.markdown("Quantization involves mapping a range of floating point values $[min, max]$ to a range of integer values (e.g., $[-128, 127]$ for `int8`).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Define your Data Range")
        min_val = st.slider("Min Float Value", -10.0, 0.0, -10.0, 0.1)
        max_val = st.slider("Max Float Value", 0.0, 10.0, 10.0, 0.1)
        
        st.markdown(f"**Data Range:** `[{min_val}, {max_val}]`")
        
    with col2:
        st.subheader("Quantization Parameters")
        dtype_option = st.selectbox("Target Data Type", ["int8 (-128 to 127)", "uint8 (0 to 255)"])
        
        if "int8" in dtype_option and "uint8" not in dtype_option:
            dtype = torch.int8
            q_min, q_max = -128, 127
        else:
            dtype = torch.uint8
            q_min, q_max = 0, 255
            
        # Calculate Scale and Zero Point
        tensor_dummy = torch.tensor([min_val, max_val])
        scale, zero_point = qu.get_q_scale_and_zero_point(tensor_dummy, dtype=dtype)
        
        st.metric("Scale", f"{scale:.4f}")
        st.metric("Zero Point", f"{zero_point}")
        
    st.markdown("---")
    st.subheader("Interactive Mapping")
    st.markdown("Enter a value to see how it gets quantized.")
    
    test_val = st.number_input("Test Float Value", value=(min_val + max_val)/2)
    
    # Quantize single value
    test_tensor = torch.tensor([test_val])
    q_val = qu.linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point, dtype)
    deq_val = qu.linear_dequantization(q_val, scale, zero_point)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Float", f"{test_val:.4f}")
    c2.metric("Quantized Integer", f"{q_val.item()}")
    c3.metric("Dequantized Float", f"{deq_val.item():.4f}", delta=f"{deq_val.item() - test_val:.4f} Error")
    
    # Visual Number Line
    st.markdown("#### Visual representation")
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Plot original range
    ax.plot([min_val, max_val], [0, 0], 'g-', linewidth=2, label='Float Range')
    ax.scatter([test_val], [0], color='g', s=100, zorder=5, label='Your Value')
    
    # Plot dequantized range (approximations)
    # Generate grid points to show discrete steps
    grid_int = torch.arange(q_min, q_max + 1, dtype=torch.float32)
    grid_float = qu.linear_dequantization(grid_int, scale, zero_point)
    
    # Filter points within plot view for clarity
    mask = (grid_float >= min_val - 1) & (grid_float <= max_val + 1)
    grid_float_visible = grid_float[mask]
    
    ax.scatter(grid_float_visible, np.zeros_like(grid_float_visible), color='b', alpha=0.3, s=20, marker='|', label='Quantization Steps')
    ax.scatter([deq_val.item()], [0], color='r', s=100, marker='x', zorder=5, label='Reconstructed')
    
    ax.set_yticks([])
    ax.set_title(f"Mapping: Float Range [{min_val}, {max_val}] -> Int Range [{q_min}, {q_max}]")
    ax.legend()
    st.pyplot(fig)
