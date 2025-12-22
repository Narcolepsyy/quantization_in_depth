import sys
sys.path.append('.')
import src.core.quantization as qu
import torch

print("Torch version:", torch.__version__)
t = torch.tensor([-1.0, 1.0])
s, z = qu.get_q_scale_and_zero_point(t)
print(f"Scale: {s}, Zero Point: {z}")
assert s > 0
print("Quantization Utils Verified.")
