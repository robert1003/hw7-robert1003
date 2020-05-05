import time
import math
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def _asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, epoch, epochs):
    now = time.time()
    s = now - since
    rs = s / epoch * (epochs - epoch)
    return '%s (- %s)' % (_asMinutes(s), _asMinutes(rs))

def train(teacher, student, optimizer, criterion, T, alpha, dataLoader, device):
    teacher.eval()
    student.train()

    total_num, total_hit, total_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(dataLoader):
        inputs, hard_labels = inputs.to(device), torch.LongTensor(labels).to(device)
        with torch.no_grad():
            soft_labels = teacher(inputs)

        optimizer.zero_grad()
        logits = student(inputs)
        loss = criterion(logits, hard_labels, soft_labels, T, alpha)
        loss.backward()
        optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

        print('training: {}/{}: accuracy={}, loss={}'.format(i + 1, len(dataLoader), total_hit / total_num, total_loss / total_num), end='\r')

    return total_loss / total_num, total_hit / total_num

def evaluate(teacher, student, criterion, T, alpha, dataLoader, device):
    teacher.eval()
    student.eval()

    total_num, total_hit, total_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(dataLoader):
        inputs, hard_labels = inputs.to(device), torch.LongTensor(labels).to(device)
        with torch.no_grad():
            soft_labels = teacher(inputs)

            logits = student(inputs)
            loss = criterion(logits, hard_labels, soft_labels, T, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

        print('training: {}/{}: accuracy={}, loss={}'.format(i + 1, len(dataLoader), total_hit / total_num, total_loss / total_num), end='\r')

    return total_loss / total_num, total_hit / total_num

def predict(model, dataLoader, device):
    model.eval()

    prediction = []
    for i, (inputs, labels) in enumerate(dataLoader):
        inputs, hard_labels = inputs.to(device), torch.LongTensor(labels).to(device)

        with torch.no_grad():
            logits = model(inputs)

        prediction.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    return np.hstack(prediction)


def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

