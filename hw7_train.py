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
from _utils import loss_fn_kd, timeSince, train, evaluate, encode8, decode8

# set random seed
random.seed(1003)
np.random.seed(1003)
torch.manual_seed(1003)
torch.cuda.manual_seed_all(1003)
torch.backends.cudnn.deterministic = True

# hyperparams
args = {
    'device': 'cuda',
    'dir': 'food-11',
    'teacher_model': './teacher_resnet18.bin',
    'BATCH_SIZE': 32,
    'log_file': 'hw7_train.log',
    'model_file': 'valid_student_model.bin',
    'model_file_8bit': 'valid_student_model_8bit.pkl',
}
args = argparse.Namespace(**args)

# load datset
train_dataloader = get_dataloader(args.dir, 'training', batch_size=args.BATCH_SIZE)
valid_dataloader = get_dataloader(args.dir, 'validation', batch_size=args.BATCH_SIZE)

# create model and optimizer
teacher_net = models.resnet18(pretrained=False, num_classes=11).to(args.device)
student_net = StudentNet(base=16).to(args.device)
teacher_net.load_state_dict(torch.load(args.teacher_model))
optimizer = optim.Adam(student_net.parameters(), amsgrad=True)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(args.log_file, 'w'), logging.StreamHandler(sys.stdout)]
)

# train!
val_best_acc = 0
epochs = 1000
start = time.time()
for epoch in range(epochs):
    logging.info('epoch {}/{}'.format(epoch + 1, epochs))
    train_loss, train_acc = train(teacher_net, student_net, optimizer, loss_fn_kd, 20, 0.5, train_dataloader, args.device)
    logging.info('train accuracy={}, train loss={}'.format(train_acc, train_loss) + ' ' * 30)
    valid_loss, valid_acc = evaluate(teacher_net, student_net, loss_fn_kd, 20, 0.5, valid_dataloader, args.device)
    logging.info('valid accuracy={}, valid loss={}'.format(valid_acc, valid_loss) + ' ' * 30)

    if valid_acc > val_best_acc:
        logging.info('update valid_best_acc: {} -> {}'.format(val_best_acc, valid_acc))
        val_best_acc = valid_acc
        torch.save(student_net.state_dict(), args.model_file)

    print(timeSince(start, epoch + 1, epochs))

# convert to 8-bit model
student_net.load_state_dict(torch.load(args.model_file))
encode8(student_net.state_dict(), args.model_file_8bit)
print(f'8-bit cost: {os.stat(args.model_file_8bit).st_size} bytes.')
