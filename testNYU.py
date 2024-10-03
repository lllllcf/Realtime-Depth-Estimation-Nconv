from dataset.nyuloader import *
from models.BPNet import BilateralMLP, nvonvDNET, CNN
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
import sys
# from models.nocuda import SIMPLE
from models.step2cutmargin import STEP2
from models.nocudaaddcspnwithconfidence import STEP3
# from models.smallModel import SMALL_STEP1, SMALL_STEP2

from models.smallModelTwoInput import SMALL_STEP2


# test loader
test_dataset = DataLoader_NYU_test('/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print('Test size: ' + str(len(test_loader)))


device = torch.device('cuda')
# best_model = nvonvDNET() # BilateralMLP()
best_model = SMALL_STEP2()
# checkpoint = torch.load("./checkpoints/buc50norelu.pth.tar")
checkpoint = torch.load("./checkpoints/epoch=100.checkpoint.pth.tar")

state_dict = checkpoint["state_dict"]

new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
best_model.load_state_dict(new_state_dict, strict=True)

best_model.to(device)
best_model.eval()

# if torch.cuda.device_count() > 1:
#     best_model = nn.DataParallel(best_model)

print('------------------------ Start Testing ------------------------')

i = 0
loss = 0
loss_all = []
j = 0
for batch, data in enumerate(test_loader):
    # if (j != int(sys.argv[1])):
    #     j = j + 1
    #     continue
    # j = j + 1
    if (batch % 10 == 0 and batch != 0):
        print('Batch No. {0}'.format(batch))

    rgb = data['rgb'].to(device)
    depth = data['depth'].to(device)
    k = data['k'].to(device)

    

    # r = 10
    # if (batch == 0):
    #     r = 1
    
    r = 1
    start = time.time()
    for q in range(r):
        estimated_depth, _ = best_model(rgb, depth, rgb, depth)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start:.5f} seconds")

    for i in range(estimated_depth.shape[0]):
        index = batch * estimated_depth.shape[0] + i
        file_path = os.path.join('./nyuresults', f'{index:010d}.png')

        depth_data = (estimated_depth[0, 0, :, :]).detach().cpu().numpy()

        # length = 480 * 640
        # fp = './nyuresults/d' + str(batch)
        # with open(fp, 'wb') as file:
        #     file.write(length.to_bytes(4, byteorder='little', signed=True))
        #     depth_data.tofile(file)

        depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        depth_data_8bit = depth_data_normalized.astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_data_8bit, cv2.COLORMAP_INFERNO)

        raw_depth_data = (depth[i, 0, :, :]).detach().cpu().numpy()
        raw_depth_data_normalized = cv2.normalize(raw_depth_data, None, 0, 255, cv2.NORM_MINMAX)
        raw_depth_data_8bit = raw_depth_data_normalized.astype(np.uint8)
        colored_raw_depth = cv2.applyColorMap(raw_depth_data_8bit, cv2.COLORMAP_INFERNO)

        rgb_data = np.transpose((rgb[i, :, :, :]).detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

        colored_depth = np.rot90(colored_depth, k=-1)
        colored_raw_depth = np.rot90(colored_raw_depth, k=-1)
        rgb_data = np.rot90(rgb_data, k=-1)

        concatenated_image = cv2.hconcat([colored_raw_depth, colored_depth, rgb_data])
        cv2.imwrite(file_path, concatenated_image)
        # cv2.imwrite(file_path, colored_depth)