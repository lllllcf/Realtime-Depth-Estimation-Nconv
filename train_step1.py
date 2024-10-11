from dataset.nyuloader import *
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

output_name = "NewLossWithMask4.0"
num_train_epoch = 50
learning_rate = [1e-4]
weight_decay = [1e-7]
apply_mask = True
add_noise = False
use_gradient_loss = True

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
    loss_train = []
    for epoch in range(num_epoch):
        loss_train = []
        for batch, data in enumerate(train_loader):
            # if (batch > 600):
            #     break
            if (batch % 100 == 0 and batch != 0):
                print('Batch No. {0}'.format(batch))

                save_depth((estimated_depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_output.png')
                save_depth((depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_sparse.png')
                save_depth((gt[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_gt.png')
                # save_depth((confidence[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_confidence.png')

            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            gt = data['gt'].to(device)
            k = data['k'].to(device)

            num_itration += 1

            model.train()
            optim.zero_grad()
            estimated_depth = model(depth, depth)

            loss = calculate_loss(estimated_depth[0, :, :, :], gt[0, :, :, :], use_gradient_loss)
            loss.requires_grad_().backward()
            optim.step()

            loss_all.append(loss.item())
            loss_train.append(loss.item())
            loss_index.append(num_itration)

        print('Epoch No. {0} -- loss = {1:.4f}'.format(
            epoch + 1,
            loss.item(),
        ))

        # Validation:
        print('Validation')
        val_loss = get_performance(model, val_loader, device_str, use_gradient_loss)
        model.to(device)
        print("Validation loss: {:.4f}".format(val_loss))
        # val_loss = sum(loss_train) / len(loss_train)
        # print("Train loss: {:.4f}".format(val_loss))

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
    print("Best validation loss: {:.4f}".format(best_val_loss))
    print('------------------------ Training Done ------------------------')
    stats = {'loss': loss_all,
             'loss_ind': loss_index,
             }

    return best_model, best_val_loss, stats
    
def get_hyper_parameters(lr, wd):
    _para_list = [{"optim_type": 'adam', 'lr': lr, "weight_decay": wd, "store_img_training": True}]
    _num_epoch = num_train_epoch
    _patience = 5
    _device = 'cuda'
    return _para_list, _num_epoch, _patience, _device


best_val_loss = float('inf')
best_model = SETP1_NCONV()
best_lr = 0
best_wd = 0
final_stats = {}
for lr in learning_rate:
    for wd in weight_decay:
        train_dataset = DataLoader_NYU('/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 'train', apply_mask, add_noise)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataset = DataLoader_NYU('/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 'val', apply_mask, add_noise)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        print('Train size: ' + str(len(train_loader)))
        print('Val size: ' + str(len(val_loader)))  
        print('Learning Rate: ' + str(lr))
        print('Weight Decay: ' + str(wd))  

        model = SETP1_NCONV()
        model = nn.DataParallel(model)
        para_list, num_epoch, patience, device_str = get_hyper_parameters(lr, wd)

        new_model, val_loss, stats = train_model(model, train_loader, val_loader, num_epoch, para_list[0], patience, device_str)

        if (val_loss < best_val_loss):
            best_model = copy.deepcopy(new_model)
            best_val_loss = val_loss
            best_lr = lr
            best_wd = wd
            final_stats = stats

print('------------------------ Training Done ------------------------')
print('---------------------------------------------------------------')
print("Best validation loss(ALL): {:.4f}".format(best_val_loss))
print("Best learning rate(ALL): {:.4f}".format(best_lr))
print("Best weight decay(ALL): {:.4f}".format(best_wd))
print('---------------------------------------------------------------')
print('------------------------ Training Done ------------------------')
save_checkpoint(best_model, num_train_epoch, "./checkpoints", final_stats, output_name)
