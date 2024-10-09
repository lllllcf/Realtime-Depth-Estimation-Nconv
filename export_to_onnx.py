import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.nyuloader import *
from models.step2 import SETP2_BP_TRAIN

from utils import *
from torch import nn
import numpy as np
import torch.optim as optimizer
import cv2
import torch
import torch.nn.functional as F
import time
import copy
import matplotlib.pyplot as plt
from models.step2 import SETP2_BP_EXPORT

step2_checkpoint_name = "test2"
output_onnx_name = "test"


model = SETP2_BP_EXPORT()
model.eval()
checkpoint = torch.load("./checkpoints/{}.pth.tar".format(step2_checkpoint_name))

state_dict = checkpoint["state_dict"]

new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

device_str = 'cuda'
device = torch.device('cuda')
model.to(device)

total_params = sum(p.numel() for p in model.parameters())

print(f'Total trainable parameters: {total_params}')

# Dummy inputs for the model
dummy_rgb = torch.randn(1, 3, 480, 640, device=device_str)   # Adjust the size as needed
dummy_depth = torch.randn(1, 1, 480, 640, device=device_str) # Adjust the size as needed
dummy_k = torch.randn(1, 3, 3, device=device_str)            # Adjust the size as needed

# onnx_program = torch.onnx.dynamo_export(model, (dummy_rgb, dummy_depth, dummy_k))
# onnx_program.save("./NEWnconvCSPN.onnx")

# import onnx
# onnx_model = onnx.load("./NEWnconvCSPN.onnx")
# onnx.checker.check_model(onnx_model)

# Export the model
onnx_model_path = ("./onnx/{}.onnx".format(output_onnx_name))   
torch.onnx.export(
    model,  # Pass the actual model if using nn.DataParallel
    (dummy_rgb, dummy_depth, dummy_rgb, dummy_depth),  # Pass the inputs as a tuple
    onnx_model_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['rgb_0', 'depth_0', 'rgb_1', 'depth_1'],
    output_names=['output_depth_0', 'output_depth_1'],
    dynamic_axes={'rgb_0': {0: 'batch_size'}, 
    'depth_0': {0: 'batch_size'}, 
    'rgb_1': {0: 'batch_size'}, 
    'depth_1': {0: 'batch_size'}, 
    'output_depth_0': {0: 'batch_size'},
    'output_depth_1': {0: 'batch_size'}
    }
)

# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)