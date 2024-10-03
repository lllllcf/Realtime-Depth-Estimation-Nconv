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

def train_model(model, train_loader, val_loader, num_epoch, parameter, patience, device_str):
    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    loss_all, loss_index = [], []
    num_itration = 0
    best_model, best_val_loss = model, float('inf')
    num_bad_epoch = 0

    optim = get_optimizer(model, parameter["optim_type"], parameter["lr"], parameter["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0, total_iters=num_epoch)

    print('------------------------ Start Training ------------------------')
    t_start = time.time()
    loss = 0
    for epoch in range(num_epoch):
        for batch, data in enumerate(train_loader):
            if (batch % 100 == 0 and batch != 0):
                print('Batch No. {0}'.format(batch))

                save_depth((estimated_depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_output.png')
                save_depth((depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_sparse.png')
                save_depth((gt[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_gt.png')

                d = (estimated_depth[0, 0, :, :] * 256.0).detach().cpu().numpy()
                cv2.imwrite("./tmp/output.png", d)
                s = (depth[0, 0, :, :] * 256.0).detach().cpu().numpy()
                cv2.imwrite("./tmp/sparse.png", s)
                g = (gt[0, 0, :, :] * 256.0).detach().cpu().numpy()
                cv2.imwrite("./tmp/gt.png", g)
                r = (rgb[0, 0, :, :]).detach().cpu().numpy()
                cv2.imwrite("./tmp/rgb.png", r)

            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            gt = data['gt'].to(device)
            k = data['k'].to(device)

            num_itration += 1

            model.train()
            optim.zero_grad()
            estimated_depth = model(rgb, depth, k)

            loss = calculate_loss(estimated_depth, gt)
            loss.requires_grad_().backward()
            optim.step()

            loss_all.append(loss.item())
            loss_index.append(num_itration)

        print('Epoch No. {0} -- loss = {1:.4f}'.format(
            epoch + 1,
            loss.item(),
        ))

        # Validation:
        print('Validation')
        val_loss = get_performance(model, val_loader, device_str)
        model.to(device)
        print("Validation loss: {:.4f}".format(val_loss))

        if val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if num_bad_epoch >= patience:
            break

        # learning rate scheduler
        scheduler.step()

    t_end = time.time()
    print('Training lasted {0:.2f} minutes'.format((t_end - t_start) / 60))
    print('------------------------ Training Done ------------------------')
    stats = {'loss': loss_all,
             'loss_ind': loss_index,
             }

    return best_model, stats
    
def get_hyper_parameters():
    _para_list = [{"optim_type": 'adam', 'lr': 0.001, "weight_decay": 1e-4, "store_img_training": True}]
    _num_epoch = 20
    _patience = 5
    _device = 'cuda'
    return _para_list, _num_epoch, _patience, _device


train_dataset = DataLoader_KITTI('../scratch/kitti_raw', 'train')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = DataLoader_KITTI('../scratch/kitti_raw', 'val')
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

print('Train size: ' + str(len(train_loader)))
print('Val size: ' + str(len(val_loader)))

model = BilateralMLP()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
para_list, num_epoch, patience, device_str = get_hyper_parameters()

best_model, stats = train_model(model, train_loader, val_loader, num_epoch, para_list[0], patience, device_str)

# save checkpoints
save_checkpoint(best_model, 40, "./checkpoints", stats)
# Plotting
plt.figure(figsize=(10, 6)) # Sets the figure size
plt.plot(stats['loss_ind'], stats['loss'], marker='o', linestyle='-', color='b')
plt.title('Loss over Time') # Title of the plot
plt.xlabel('Index') # X-axis label
plt.ylabel('Loss') # Y-axis label
plt.grid(True) # Shows a grid
plt.savefig('./statsKITTI.png') # Saves the plot as a .png file