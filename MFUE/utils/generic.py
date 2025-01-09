#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import os
import sys
import logging
import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import timm
import copy
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import random
import models
from . import data
from . import generate


def random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def train_model(model, args):
    data_file = torch.load(args.data_path + '/' + args.dataset + '/allocated_data/data_log.pth')
    
    train_ldr = DataLoader(TensorDataset(data_file["X_train"], data_file["y_train"]), 
                           batch_size = args.batch_size, shuffle=True)

    test_ldr = DataLoader(TensorDataset(data_file["X_remain"], data_file["y_remain"]), 
                           batch_size = 128, shuffle=False)
    
    # define the optimizer
    optimizer = get_optim(args, model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[40, 80, 120], 
                                                     gamma=0.2)
    # train the source_model
    model.to(args.device)
    
    model_dir = args.model_path + '/' + args.dataset + '/' + args.arch + '/'
    os.makedirs(model_dir, exist_ok=True)
    logger = get_logger(model_dir + 'log.txt')
    logger.info('start training!')
    for epoch in range(args.epochs):
        print("epoch=",epoch)
        model.train()
        loss_meter = 0
        acc_meter = 0
        runcount = 0
        
        for batch_idx, (x, y) in enumerate(train_ldr):
            optimizer.zero_grad()
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            runcount += x.size(0)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            
            pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            acc_meter += pred.eq(y.view_as(pred)).sum().item()
            loss_meter += (loss.item()*x.size(0))

        loss_meter /= runcount
        acc_meter /= (runcount/100)
        
        # adjust the scheduler 
        scheduler.step()
        
        # calculate test acc
        test_acc_meter = test_acc_dataldr(model, args.device, test_ldr)
        
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t train_acc={:.3f} test_acc={:.3f}'.format(epoch , args.epochs, loss_meter, acc_meter, test_acc_meter ))
        

        
    w = copy.deepcopy(model.state_dict())
    
    torch.save(w, model_dir + 'model.pth')
    
    return model




def test_acc_dataldr(model, device, data_ldr):
    test_acc_meter = 0
    runcount = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_ldr):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs).argmax(dim=1)
            test_acc_meter += (pred==targets).sum().item()
            runcount += inputs.size(0)

    test_acc_meter /= (runcount/100)
    return test_acc_meter

def test_metric_dataldr(model, device, data_ldr):
    test_acc_meter = 0
    train_loss_meter = 0
    pred_unc_meter = 0
    grad_norm_meter = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.0)

    runcount = 0
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(data_ldr):
        inputs, targets = inputs.to(device), targets.to(device)
        pred = model(inputs)
        
        # loss
        loss = criterion(pred, targets)
        train_loss_meter += loss.item()
        # prediction uncertainty
        pred_unc_meter += torch.norm(pred, dim=0).sum().item()
        # prediction Acc
        y_pred = pred.argmax(dim=1)
        test_acc_meter += (y_pred==targets).sum().item()
        
        runcount += inputs.size(0)
        
        # gradient norm
        optim.zero_grad()
        loss.backward()
        for pp in model.parameters():
            grad_norm_meter += torch.norm(pp.grad).item()
    
    record = [  test_acc_meter,
                train_loss_meter,
                pred_unc_meter,
                grad_norm_meter]
    
    for i in range(len(record)):
        record[i] = record[i]/(runcount/100)
    
    return record

def test_loss_dataldr(model, device, data_ldr):
    train_loss_meter = 0
    criterion = torch.nn.CrossEntropyLoss()

    runcount = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_ldr):
        inputs, targets = inputs.to(device), targets.to(device)
        pred = model(inputs)
        # prediction Acc
        y_pred = pred.argmax(dim=1)
        selected_idex = y_pred==targets
        # loss
        loss = criterion(pred[selected_idex], targets[selected_idex])
        train_loss_meter += loss.item()
        
        runcount += inputs.size(0)
        
    train_loss_meter = train_loss_meter / (runcount/100)
    
    return train_loss_meter


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def get_pretrained_model(args, specify=None):
    model = get_arch(args.arch, args.dataset)
    if specify == None:
        model_dir = args.model_path + '/' + args.dataset + '/' + args.arch + '/model.pth'
    else:
        model_dir = specify
    model.load_state_dict(torch.load(model_dir,
                          map_location=args.device))
    return model


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_transforms(dataset, train=True, is_tensor=True):

    if train:
        if dataset == 'cifar10' or dataset == 'cifar100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), ]
        elif dataset == 'tiny-imagenet':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 8), ]
        else:
            raise NotImplementedError
    else:
        comp1 = []

    if is_tensor:
        comp2 = [
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = data.ElementWiseTransform(trans)

    return trans


def get_filter(fitr):
    if fitr == 'averaging':
        return lambda x: cv2.blur(x, (3,3))
    elif fitr == 'gaussian':
        return lambda x: cv2.GaussianBlur(x, (3,3), 0)
    elif fitr == 'median':
        return lambda x: cv2.medianBlur(x, 3)
    elif fitr == 'bilateral':
        return lambda x: cv2.bilateralFilter(x, 9, 75, 75)

    raise ValueError


def get_dataset(dataset, root='./data', train=True, fitr=None):
    transform = get_transforms(dataset, train=train, is_tensor=False)
    lp_fitr   = None if fitr is None else get_filter(fitr)

    if dataset == 'cifar10':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'tiny-imagenet':
        target_set = data.datasetTinyImageNet(root=root, train=train, transform=transform)
        x, y = target_set.x, target_set.y
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    return data.Dataset(x, y, transform, lp_fitr)



def get_arch(arch, dataset):
    if dataset == 'cifar10':
        in_dims, out_dims = 3, 10
        img_size = 32
        patch_size = 4
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
        img_size = 32
        patch_size = 4
    elif dataset == 'tiny-imagenet':
        in_dims, out_dims = 3, 200
        img_size = 64
        patch_size = 16
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    if arch == 'resnet18':
        return models.resnet18(in_dims, out_dims)

    elif arch == 'resnet50':
        return models.resnet50(in_dims, out_dims)

    elif arch == 'wrn-34-10':
        return models.wrn34_10(in_dims, out_dims)

    elif arch == 'vgg11-bn':
        return models.vgg11_bn(in_dims, out_dims)

    elif arch == 'vgg16-bn':
        return models.vgg16_bn(in_dims, out_dims)

    elif arch == 'vgg19-bn':
        return models.vgg19_bn(in_dims, out_dims)

    elif arch == 'densenet-121':
        return models.densenet121(num_classes=out_dims)
    
    elif arch == 'vit':
        if dataset == 'cifar10' or dataset == 'cifar100':
            return models.ViT(
                image_size=img_size, patch_size=patch_size, num_classes=out_dims, dim=512,
                depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1
            )
            
        
        else:
            raise NotImplementedError('architecture {} on {} is not supported'.format(arch, dataset))
    
    elif arch == 'swin':
        
        if dataset == 'cifar10' or dataset == 'cifar100':
            return models.swin_t(
                window_size=patch_size, num_classes=out_dims, downscaling_factors=(2,2,2,1)
            )
        
        else:
            raise NotImplementedError('architecture {} on {} is not supported'.format(arch, dataset))

    else:
        raise NotImplementedError('architecture {} is not supported'.format(arch))


def get_optim(args, params):
    if args.optim == 'sgd':
        return torch.optim.SGD(params, lr=args.lr,
                                             momentum=args.momentum,
                                             weight_decay = args.weight_decay,
                                             nesterov = True)
    elif args.optim == 'adam':
        return torch.optim.Adam(params, args.lr, weight_decay = args.weight_decay)

    raise NotImplementedError('optimizer {} is not supported'.format(args.optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    return logger


def evaluate(model, criterion, loader, cpu):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    for x, y in loader:
        if not cpu: x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()


