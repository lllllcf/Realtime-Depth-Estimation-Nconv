from dataset.kittiloader import *
from models.BPNet import BilateralMLP
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
import time

# test loader
test_dataset = DataLoader_KITTI_test('/oscar/data/jtompki1/cli277/kitti_raw')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print('Test size: ' + str(len(test_loader)))

device = torch.device('cuda')
best_model = BilateralMLP()
checkpoint = torch.load("./checkpoints/epoch=40.checkpoint.pth.tar")

state_dict = checkpoint["state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
best_model.load_state_dict(new_state_dict)

best_model.to(device)
best_model.eval()

if torch.cuda.device_count() > 1:
    best_model = nn.DataParallel(best_model)

print('------------------------ Start Testing ------------------------')

i = 0
loss = 0
loss_all = []
for batch, data in enumerate(test_loader):
    if (batch % 100 == 0 and batch != 0):
        print('Batch No. {0}'.format(batch))

    rgb = data['rgb'].to(device)
    depth = data['depth'].to(device)
    k = data['k'].to(device)

    estimated_depth = best_model(rgb, depth, k)

    start = time.time()
    for i in range(estimated_depth.shape[0]):
        index = batch * estimated_depth.shape[0] + i
        file_path = os.path.join('./results', f'{index:010d}.png')
        save_depth((estimated_depth[i, 0, :, :]).detach().cpu().numpy(), file_path)
    
    end_time = time.time()

    print(f"Time taken: {end_time - start:.2f} seconds")