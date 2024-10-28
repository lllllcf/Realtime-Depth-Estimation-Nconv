import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os
import random
import dataset.data_utils as data_utils

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
            pose_path = os.path.join(data_dir, 'void_1500/test_absolute_pose.txt')
            gt_path = os.path.join(data_dir, 'void_1500/test_ground_truth.txt')
            rgb_path = os.path.join(data_dir, 'void_1500/test_image.txt')
            intrinsics_path = os.path.join(data_dir, 'void_1500/test_intrinsics.txt')
            sparse_depth_path = os.path.join(data_dir, 'void_1500/test_sparse_depth.txt')
            validity_map_path = os.path.join(data_dir, 'void_1500/test_validity_map.txt')

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
            sample = {'pose': pose, 'rgb': rgb, 'depth': sparse_depth.unsqueeze(0), 'gt': gt.unsqueeze(0), 'k': k}
            return sample
        else:
            sparse_depth = self.preprocess_depth(self.sparse_depths[index], self.use_mask)
            sample = {'pose': pose, 'rgb': rgb, 'depth': sparse_depth.unsqueeze(0), 'gt': gt.unsqueeze(0), 'k': k}
            return sample
    
    def preprocess_depth(self, depth_path, apply_mask):
        depth = self.get_sparse_depth(depth_path)

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
        return torch.FloatTensor(data_utils.load_depth(depth_path))
    
    def get_gt(self, gt_path):
        return torch.FloatTensor(data_utils.load_depth(gt_path))

    def get_K(self, k_path):
        return torch.FloatTensor(np.loadtxt(k_path))
