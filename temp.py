import torch
import torch.nn as nn
import time
from models.BPNet import BilateralMLP

# Initialize the network
net = BilateralMLP()

# Print parameter sizes and total number of parameters
total_params = 0
for name, param in net.named_parameters():
    if param.requires_grad:
        param_size = param.numel()
        total_params += param_size
        print(f"Parameter: {name}, Size: {param.size()}, Number of elements: {param_size}")

print(f"Total number of parameters: {total_params}")

# Create random input tensors
fout = torch.randn(1, 3, 100, 100)
S = torch.randn(1, 1, 100, 100)
K = torch.randn(1, 3, 3)

# Run the forward method
start_time = time.time()
output = net.forward(fout, S, K)
end_time = time.time()
print(f"Forward pass took {end_time - start_time:.6f} seconds")

print(output.shape)