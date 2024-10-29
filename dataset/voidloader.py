import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os
import random
import dataset.data_utils as data_utils

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class DataLoader_VOID(Dataset):
    def __init__(self, data_dir, mode,  use_mask, height=480, width=640):

        if (mode == 'train'):
            pose_path = os.path.join(data_dir, 'void_1500/train_absolute_pose.txt')
            gt_path = os.path.join(data_dir, 'void_1500/train_ground_truth.txt')
            rgb_path = os.path.join(data_dir, 'void_1500/train_image.txt')
            intrinsics_path = os.path.join(data_dir, 'void_1500/train_intrinsics.txt')
            sparse_depth_path = os.path.join(data_dir, 'void_1500/train_sparse_depth.txt')
            validity_map_path = os.path.join(data_dir, 'void_1500/train_validity_map.txt')
        elif (mode == 'val'):
            pose_path = os.path.join(data_dir, 'void_1500/val_absolute_pose.txt')
            gt_path = os.path.join(data_dir, 'void_1500/val_ground_truth.txt')
            rgb_path = os.path.join(data_dir, 'void_1500/val_image.txt')
            intrinsics_path = os.path.join(data_dir, 'void_1500/val_intrinsics.txt')
            sparse_depth_path = os.path.join(data_dir, 'void_1500/val_sparse_depth.txt')
            validity_map_path = os.path.join(data_dir, 'void_1500/val_validity_map.txt')

        self.poses = data_utils.read_paths(data_dir, pose_path)
        self.sparse_depths = data_utils.read_paths(data_dir, sparse_depth_path)
        self.gts = data_utils.read_paths(data_dir, gt_path)
        self.rgbs = data_utils.read_paths(data_dir, rgb_path)
        self.Ks = data_utils.read_paths(data_dir, intrinsics_path)

        mask_path = os.path.join(data_dir, 'void_1500/mask')
        self.masks = list(sorted(glob.iglob(mask_path + "/*.npy", recursive=True)))

        self.use_mask = use_mask


    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item(self, index):
        pose = self.get_pose(self.poses[index])
        rgb = self.get_rgb(self.rgbs[index])
        # sparse_depth = self.get_sparse_depth(self.sparse_depths[index])
        gt = self.get_gt(self.gts[index])
        k = self.get_K(self.Ks[index])

        if (self.use_mask):
            sparse_depth = self.preprocess_depth(self.gts[index], self.use_mask)
            sample = {'pose': pose, 'rgb': rgb, 'depth': sparse_depth, 'gt': self.edge_inpainting(gt), 'k': k}
            return sample
        else:
            sparse_depth = self.preprocess_depth(self.sparse_depths[index], self.use_mask)
            sample = {'pose': pose, 'rgb': rgb, 'depth': sparse_depth, 'gt': self.edge_inpainting(gt), 'k': k}
            return sample
    
    def edge_inpainting(self, input_depth):

        depth = input_depth
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)  # Add channel dimension if missing

        depth = depth.unsqueeze(0)  # Add batch dimension: Shape (1, C, H, W)

        # --- Edge Detection using Sobel Operator in PyTorch ---
        sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]]], dtype=torch.float32).to(depth.device)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]]], dtype=torch.float32).to(depth.device)

        sobel_kernel_x = sobel_kernel_x.unsqueeze(0)  # Shape: (1, 1, 3, 3)
        sobel_kernel_y = sobel_kernel_y.unsqueeze(0)  # Shape: (1, 1, 3, 3)

        # Compute gradients along the X and Y axes
        grad_x = F.conv2d(depth, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_kernel_y, padding=1)

        # Compute the gradient magnitude (edge strength)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        edge_magnitude_norm = edge_magnitude.squeeze(0).squeeze(0)  # Shape: (H, W)

        threshold_value = 0.5  # Adjust this value as needed
        edge_map = (edge_magnitude_norm > threshold_value).float()  # Binary edge map

            # --- Edge Removal ---
        non_edge_mask = 1.0 - edge_map  # Invert edge map
        depth_no_edges = depth.squeeze(0) * non_edge_mask  # Apply mask to depth map

        depth_no_edges = depth_no_edges.unsqueeze(0)  # Add batch dimension back

        # return depth_no_edges  # Return the inpainted depth map with batch dimension

            # --- Inpainting with Nearest Neighbor after Edge Removal ---
        non_edge_mask = (edge_map).numpy().astype(np.uint8)  # Convert mask to uint8
        depth_no_edges = depth.squeeze(0).squeeze(0).numpy()  # Convert tensor to numpy

        # inpainted_depth = cv2.inpaint(depth_no_edges, non_edge_mask, 3, cv2.INPAINT_TELEA)
        inpainted_depth = self.inpaint_with_nearest(depth_no_edges, non_edge_mask)

        return torch.tensor(inpainted_depth).unsqueeze(0)
    
    def inpaint_with_nearest(self, depth_map, mask):
        # Convert the mask to binary (0 or 255)
        mask = (mask * 255).astype(np.uint8)

        # Use dilation to expand the nearest pixel values into the masked area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inpainted = depth_map.copy()
        for i in range(5):  # Increase the number of iterations for larger holes
            inpainted[mask == 255] = cv2.dilate(inpainted, kernel)[mask == 255]

        return inpainted
    
    def preprocess_depth(self, depth_path, apply_mask):
        depth = self.get_sparse_depth(depth_path)
        depth = self.edge_inpainting(depth)

        if apply_mask:
            mask_path = random.choice(self.masks)
            mask_raw = np.load(mask_path)
            
            if mask_raw.shape != (480, 640):
                mask_image = Image.fromarray(mask_raw)
                mask_image = mask_image.resize((640, 480), Image.NEAREST)  # Using nearest neighbor to avoid interpolation of binary values
                mask = np.array(mask_image)
            else:
                mask = mask_raw

            depth = depth * torch.FloatTensor(mask) 

        return depth

    def get_pose(self, pose_path):
        return torch.FloatTensor(np.loadtxt(pose_path))
    
    def get_rgb(self, rgb_path):
        return torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)
    
    def get_sparse_depth(self, depth_path):
        return torch.FloatTensor(data_utils.load_depth(depth_path)).unsqueeze(0)
    
    def get_gt(self, gt_path):
        return torch.FloatTensor(data_utils.load_depth(gt_path)).unsqueeze(0)

    def get_K(self, k_path):
        return torch.FloatTensor(np.loadtxt(k_path))
