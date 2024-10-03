import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class DataLoader_KITTI(Dataset):
    def __init__(self, data_dir, mode, height=256, width=1216, tp_min=50):

        self.depth_path = os.path.join(data_dir, 'data_depth_annotated', mode)
        self.lidar_path = os.path.join(data_dir, 'data_depth_velodyne', mode)
        self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
        self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))

        self.height = height
        self.width = width
        self.tp_min = tp_min

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item(self, index):
        file_names = self.depths[index].split('/')
        rgb_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0], file_names[-5],
                                    file_names[-2], 'data', file_names[-1])

        rgb = self.get_rgb(rgb_path)
        depth = self.get_depth(self.lidars[index])
        gt = self.get_gt(self.depths[index])
        k = self.get_k(index)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        gt = gt[:, tp:tp + self.height, lp:lp + self.width]
        k[0, 2] -= lp
        k[1, 2] -= tp

        sample = {'rgb': rgb, 'depth': depth, 'gt': gt, 'k': k}
        return sample
    
    def get_rgb(self, rgb_path):
        return torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)
    
    def get_depth(self, depth_path):
        d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 256.0
        exp_d = np.expand_dims(d, axis=0)
        depth = torch.FloatTensor(exp_d)
        return depth
    
    def get_gt(self, gt_path):
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 256.0
        gt_exp = np.expand_dims(gt_raw, axis=0)
        gt = torch.FloatTensor(gt_exp)
        return gt
    
    def get_k(self, index):
        file_names = self.depths[index].split('/')

        calib_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0],
                                    'calib_cam_to_cam.txt')
        filedata = read_calib_file(calib_path)
        P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))
        if file_names[-2] == 'image_02':
            K_cam = P_rect_20[0:3, 0:3]
        elif file_names[-2] == 'image_03':
            K_cam = P_rect_30[0:3, 0:3]
        else:
            raise ValueError("Unknown mode: {}".format(file_names[-2]))
        
        return torch.FloatTensor(np.array(K_cam).astype(np.float32).reshape(3, 3))


class DataLoader_KITTI_seltest(Dataset):
    def __init__(self, data_dir, height=256, width=1216, tp_min=50):

        self.depth_path = os.path.join(data_dir, 'val_selection_cropped', 'groundtruth_depth')
        self.lidar_path = os.path.join(data_dir, 'val_selection_cropped', 'velodyne_raw')
        self.image_path = os.path.join(data_dir, 'val_selection_cropped', 'image')
        self.depths = list(sorted(glob.iglob(self.depth_path + "/*.png", recursive=True)))
        self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
        self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))

        self.height = height
        self.width = width
        self.tp_min = tp_min

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item(self, index):
        rgb_path = self.images[index]

        rgb = self.get_rgb(rgb_path)
        depth = self.get_depth(self.lidars[index])
        gt = self.get_gt(self.depths[index])
        k = self.get_k(index)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        gt = gt[:, tp:tp + self.height, lp:lp + self.width]
        k[0, 2] -= lp
        k[1, 2] -= tp

        sample = {'rgb': rgb, 'depth': depth, 'gt': gt, 'k': k}
        return sample
    
    def get_rgb(self, rgb_path):
        return torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)
    
    def get_depth(self, depth_path):
        d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 256.0
        exp_d = np.expand_dims(d, axis=0)
        depth = torch.FloatTensor(exp_d)
        return depth
    
    def get_gt(self, gt_path):
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 256.0
        gt_exp = np.expand_dims(gt_raw, axis=0)
        gt = torch.FloatTensor(gt_exp)
        return gt
    
    def get_k(self, index):
        fns = self.images[index].split('/')
        calib_path = os.path.join(*fns[:-2], 'intrinsics', fns[-1][:-3] + 'txt')
        with open(calib_path, 'r') as f:
            K_cam = f.read().split()
        
        return torch.FloatTensor(np.array(K_cam).astype(np.float32).reshape(3, 3))


class DataLoader_KITTI_test(Dataset):
    def __init__(self, data_dir, height=352, width=1216, tp_min=50):

        self.lidar_path = os.path.join(data_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
        self.image_path = os.path.join(data_dir, 'test_depth_completion_anonymous', 'image')
        self.lidars = list(sorted(glob.iglob('/' + self.lidar_path + "/*.png", recursive=True)))
        self.images = list(sorted(glob.iglob('/' + self.image_path + "/*.png", recursive=True)))
        self.depths = self.lidars

        self.height = height
        self.width = width
        self.tp_min = tp_min

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item(self, index):
        rgb_path = self.images[index]

        rgb = self.get_rgb(rgb_path)
        depth = self.get_depth(self.lidars[index])
        k = self.get_k(index)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        k[0, 2] -= lp
        k[1, 2] -= tp

        sample = {'rgb': rgb, 'depth': depth, 'k': k}
        return sample
    
    def get_rgb(self, rgb_path):
        return torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)
    
    def get_depth(self, depth_path):
        d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 256.0
        exp_d = np.expand_dims(d, axis=0)
        depth = torch.FloatTensor(exp_d)
        return depth
    
    def get_k(self, index):
        fns = self.images[index].split('/')
        calib_path = os.path.join(*fns[:-2], 'intrinsics', fns[-1][:-3] + 'txt')
        with open('/' + calib_path, 'r') as f:
            K_cam = f.read().split()
        
        return torch.FloatTensor(np.array(K_cam).astype(np.float32).reshape(3, 3))