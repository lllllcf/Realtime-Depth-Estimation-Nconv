import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os
import random

import os
import glob
import re

# Function to extract the numeric value from the filename
def extract_number(filename):
    # Use a regular expression to find numeric parts in the filename
    match = re.search(r'\d+', os.path.basename(filename))
    if match:
        return int(match.group())  # Convert the matched number to an integer
    return float('inf')  # If no number is found, sort such files last


class TEST_DataLoader_NYU(Dataset):
    def __init__(self, data_dir, mode, height=480, width=640, tp_min=50):

        # path='/oscar/data/jtompki1/cli277/new_spot_data/1'
        path='/oscar/data/jtompki1/cli277/nyuv2/nyuv2/test'
        self.width = width
        self.height = height


        crop_size = (480, 640)
        self.crop_size = crop_size

        self.Kcam = torch.from_numpy(np.array(
            [
                [721.5377, 0, 596.5593],
                [0, 721.5377, 149.854],
                [0, 0, 1.0],
            ], dtype=np.float32
        ))

        # Define directories for color and depth files
        base_dir = path
        self.color_dir = os.path.join(base_dir, "color")
        self.depth_dir = os.path.join(base_dir, "depth")

        # List RGB and depth files
        self.rgb_list = list(sorted(glob.glob(os.path.join(self.color_dir, "*.png")), key=extract_number))
        self.depth_list = list(sorted(glob.glob(os.path.join(self.depth_dir, "*")), key=extract_number))
        print(self.depth_list[0:20])
        print('These are the total depth files being loaded' + str(len(self.depth_list)))
        # print('This is an example depth_file' + self.depth_list[3])
        # print('This is an example of extract number' + str(extract_number(self.depth_list[3])))

        self.batch_to_og_idx = {}
        for i in range(len(self.depth_list)):
            self.batch_to_og_idx[i] = extract_number(self.depth_list[i])
        
        # print('This is the map generated' + str(self.batch_to_og_idx))

        assert len(self.rgb_list) == len(self.depth_list), "Mismatch between RGB and depth file counts!"

    def __len__(self):
        return len(self.rgb_list)


    def get_original_data_idx(self):
        return self.batch_to_og_idx
        

    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = self.rgb_list[idx]
        rgb = torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)

        # Load depth map from binary file
        depth_path = self.depth_list[idx]
        
        d = self.load_depth_from_binary(depth_path)
        exp_d = np.expand_dims(d, axis=0)
        depth_map = torch.FloatTensor(exp_d)
        # print(depth_map.shape)

        # Set dep_sp to dep
        dep_sp = depth_map.clone()

        return {'rgb': rgb, 'depth': dep_sp, 'gt': dep_sp}

    def load_depth_from_binary(self, file_path):
        """
        Load a depth map from a binary file.

        Args:
            file_path (str): Path to the binary file.

        Returns:
            np.ndarray: 2D depth map.
        """
        try:
            with open(file_path, 'rb') as f:
                # Read the first 4 bytes to get the data length (int32)
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    raise ValueError(f'File {file_path} is too short to contain a valid header.')

                data_length = np.frombuffer(length_bytes, dtype=np.int32)[0]

                # Expected data length
                expected_size = self.width * self.height

                if data_length != expected_size:
                    raise ValueError(
                        f'File {file_path} has data length {data_length}, expected {expected_size}.'
                    )

                # Read the float32 depth data
                float_data = np.fromfile(f, dtype=np.float32, count=data_length)

                if len(float_data) != expected_size:
                    raise ValueError(
                        f'File {file_path} contains {len(float_data)} float values, expected {expected_size}.'
                    )

                # Reshape to 2D depth map
                depth_map = float_data.reshape((self.height, self.width))
                return depth_map
        except Exception as e:
            print(f"Error loading depth map: {e}")
            return Image.fromarray(np.zeros((self.height, self.width), dtype='float32'), mode='F')

            