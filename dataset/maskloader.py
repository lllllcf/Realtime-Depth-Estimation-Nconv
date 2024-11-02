import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os
import random
import time

class MaskLoader(Dataset):
    def __init__(self, data_dir, device, height=480, width=640, tp_min=50):
        self.mask_dir   = os.path.join(data_dir)
        self.mask_files = list(sorted(glob.iglob(self.mask_dir + "/*.npy", recursive=True)))
        
        masks = []
        for mask_file in self.mask_files:
            mask_raw = np.load(mask_file)
                    
            if mask_raw.shape != (480, 640):
                mask_image = Image.fromarray(mask_raw)
                mask_image = mask_image.resize((640, 480), Image.NEAREST)  # Using nearest neighbor to avoid interpolation of binary values
                mask = np.array(mask_image)
            else:
                mask = mask_raw

            masks.append(mask)

        self.masks = torch.tensor(masks, device=device)


    def pin_memory(self):
        breakpoint()
        self.masks  = self.masks.pin_memory()

        return self

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.masks[idx]