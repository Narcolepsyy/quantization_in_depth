import torch

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """
    Quantizes a tensor using linear quantization with given scale and zero point.
    """
    scaled_and_shift_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shift_tensor)
    
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    
    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    return q_tensor

def linear_dequantization(quantized_tensor, scale, zero_point):
    """
    Dequantizes a tensor using linear quantization logic.
    """
    return scale * (quantized_tensor.float() - zero_point)

def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    """
    Calculates Optimal Scale and Zero Point for a given tensor.
    """
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    scale = (max_val - min_val) / (q_max - q_min)
    if scale == 0:
        scale = 1.0
        
    zero_point = q_min - round(min_val / scale)
    
    # Clamp zero_point to valid range for the dtype
    zero_point = max(q_min, min(q_max, zero_point))
    
    return scale, int(zero_point)

def quantize_image(image_tensor, num_bits=8):
    """
    Quantizes an image tensor to a specific bit-width.
    Simulates lower bit-width by using a smaller range of integers.
    """
    # Assuming image is normalized or standard format.
    # We'll treat it as a float tensor to be quantized.
    
    q_min = 0
    q_max = 2**num_bits - 1
    
    min_val = image_tensor.min().item()
    max_val = image_tensor.max().item()
    
    scale = (max_val - min_val) / (q_max - q_min)
    if scale == 0:
        scale = 1.0
    
    zero_point = q_min - round(min_val / scale)
    
    # Quantize
    scaled = image_tensor / scale + zero_point
    rounded = torch.round(scaled).clamp(q_min, q_max)
    
    # Dequantize immediately to show the visual effect (fake quantization)
    dequantized = scale * (rounded - zero_point)
    
    return dequantized, scale, zero_point
