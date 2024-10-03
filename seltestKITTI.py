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

# test loader
test_dataset = DataLoader_KITTI_seltest('../scratch/kitti_raw')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print('Selvel Data Size: ' + str(len(test_loader)))

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

print('------------------------ Start Self Testing ------------------------')

i = 0
loss = 0
loss_all = []
for batch, data in enumerate(test_loader):
    if (batch % 100 == 0 and batch != 0):
        print('Batch No. {0}'.format(batch))

    rgb = data['rgb'].to(device)
    depth = data['depth'].to(device)
    gt = data['gt'].to(device)
    k = data['k'].to(device)

    estimated_depth = best_model(rgb, depth, k)
    loss = calculate_loss(estimated_depth, gt)
    loss_all.append(loss.item())

    d = (estimated_depth[0, 0, :, :]).detach().cpu().numpy()
    s = (depth[0, 0, :, :]).detach().cpu().numpy()
    g = (gt[0, 0, :, :]).detach().cpu().numpy()
    r = (rgb[0, 0, :, :]).detach().cpu().numpy()

    save_depth(s, "./test_res/" + str(i) + "_sparse.png")
    save_depth(d, "./test_res/" + str(i) + "_estimated.png")
    save_depth(g, "./test_res/" + str(i) + "_groundtruth.png")
    cv2.imwrite("./test_res/" + str(i) + "_rgb.png", r)
    i = i + 1

print("Seltest loss: {:.4f}".format(sum(loss_all) / len(loss_all)))