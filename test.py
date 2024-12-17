from dataset.nyuloader import *
from models.step2 import SETP2_BP_EXPORT
from models.step1 import SETP1_NCONV 
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
import gc
##################################################################################
#declare model architecture
step2 = SETP2_BP_EXPORT()
step1 = SETP1_NCONV()
##################################################################################

##################################################################################
#load checkpoints
step2_checkpoint_name = "without_step1"
step2.eval()
step2_checkpoint = torch.load("./checkpoints/{}.pth.tar".format(step2_checkpoint_name))
state_dict = step2_checkpoint["state_dict"]

new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
step2.load_state_dict(new_state_dict, strict=False)

print("Successfully loaded checkpoints: " + step2_checkpoint_name)

##################################################################################

##################################################################################
#load checkpoints
# step1_checkpoint_name = "Test"
# # torch.cuda.empty_cache()
# # gc.collect()
# # best_val_loss = float('inf')
# step1.eval()
# step1_checkpoint = torch.load("./checkpoints/{}.pth.tar".format(step1_checkpoint_name))
# # step2_checkpoint = torch.load("baseline2.pth.tar")
# state_dict = step1_checkpoint["state_dict"]

# new_state_dict = {}
# for k, v in state_dict.items():
#     name = k[7:] if k.startswith("module.") else k
#     new_state_dict[name] = v
# step1.load_state_dict(new_state_dict, strict=False)

# print("Successfully loaded checkpoints in step1")

##################################################################################
#send the model to GPU for processing faster
device_str = 'cuda'
device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
step2.to(device)
##################################################################################


file_index = str(2420)
alpha = str(81)

path_to_color_data = '/users/aidhant/data/aidhant/color/color_test_data' + file_index + '.png'
path_depth_data = '/users/aidhant/data/aidhant/depth/depth_test_data' + file_index + '.csv'
npy_depth_path = '/users/aidhant/data/aidhant/depth/depth_test_data' + file_index + '.npy'


# Convert CSV depth data to NPY format if it hasn't been done yet
if not os.path.exists(npy_depth_path):
    depth_data = np.loadtxt(path_depth_data, delimiter=',')  # Load CSV file
    np.save(npy_depth_path, depth_data)  # Save as .npy file

# Define the functions for loading the RGB and depth images
def get_rgb(rgb_path):
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        raise ValueError("Could not read the RGB image. Check the file path.")
    return torch.FloatTensor(rgb_image).permute(2, 0, 1)  # Change to (C, H, W)

def get_depth(depth_path):
    d = np.load(depth_path).reshape(480, 640)  # Load .npy file
    exp_d = np.expand_dims(d, axis=0)  # Add channel dimension
    return torch.FloatTensor(exp_d)  # Convert to tensor

def save_depth(depth_data, path):
    depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_data_8bit = depth_data_normalized.astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_data_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, colored_depth)

def get_gt(gt_path):
    gt_raw = np.load(gt_path).reshape(480, 640)
    gt_exp = np.expand_dims(gt_raw, axis=0)
    gt = torch.FloatTensor(gt_exp)
    return gt


# Load the RGB and depth data
rgb = get_rgb(path_to_color_data).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
depth = get_depth(npy_depth_path).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
gt = get_gt(gt_path=npy_depth_path).unsqueeze(0).to(device)

# Model outputs a depth and color image
estimated_depths, _ = step2(rgb, depth, rgb, depth)
# estimated_depths = step1(depth)

save_depth((estimated_depths[0, 0, :, :]).detach().cpu().numpy(), 'weight_variation/model_var/without_step_1/color_output' + alpha + '.png')
save_depth((depth[0, 0, :, :]).detach().cpu().numpy(), 'weight_variation/color_sparse.png')
save_depth((rgb[0, 0, :, :]).detach().cpu().numpy(), 'weight_variation/color_img.png')


# for i in range(38):
#     file_index = str(121 * i)
    
#     path_to_color_data = '/users/aidhant/data/aidhant/color/color_test_data' + file_index + '.png'
#     path_depth_data = '/users/aidhant/data/aidhant/depth/depth_test_data121' + file_index + '.csv'
#     npy_depth_path = '/users/aidhant/data/aidhant/depth/depth_test_data121' + file_index + '.npy'

#     if not os.path.exists(npy_depth_path):
#         depth_data = np.loadtxt(path_depth_data, delimiter=',')  # Load CSV file
#         np.save(npy_depth_path, depth_data)  # Save as .npy file

#     rgb = get_rgb(path_to_color_data).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
#     depth = get_depth(npy_depth_path).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
#     gt = get_gt(gt_path=npy_depth_path).unsqueeze(0).to(device)

#     estimated_depths, _ = step2(rgb, depth, rgb, depth)

#     save_depth((estimated_depths[0, 0, :, :]).detach().cpu().numpy(), 'tmp_test/color_output' + file_index + '.png')
#     save_depth((depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp_test/color_sparse' + file_index + '.png')
#     save_depth((gt[0, 0, :, :]).detach().cpu().numpy(), 'tmp_test/color_gt' + file_index + '.png')

#     print(path_to_color_data)