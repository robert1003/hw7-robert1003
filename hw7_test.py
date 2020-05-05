import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import packages
import torch
import os
import sys
import time
import argparse
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import models
from _dataset import get_dataloader
from _model import StudentNet
from _utils import loss_fn_kd, timeSince, predict, encode8, decode8

# hyperparamBATCH_SIZEs
args = {
    'device': 'cuda',
    'dir': sys.argv[1],
    'pred_file': sys.argv[2],
    'BATCH_SIZE': 32,
    'model_file_8bit': 'valid_student_model_8bit.pkl',
}
args = argparse.Namespace(**args)

# load datset
test_dataloader = get_dataloader(args.dir, 'testing', batch_size=32)

# create model and optimizer
student_net = StudentNet(base=16).to(args.device)
optimizer = optim.Adam(student_net.parameters(), amsgrad=True)

# load model
student_net.load_state_dict(decode8(args.model_file_8bit))

# get prediction
prediction = predict(student_net, test_dataloader, args.device)

# write prediction
with open(args.pred_file, 'w') as f:
    print('Id,label', file=f)
    for i, pred in enumerate(prediction):
        print(i, pred, sep=',', file=f)
    
