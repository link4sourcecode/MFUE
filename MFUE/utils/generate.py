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
import utils


class PGD_uf():
    def __init__(self,fin_num, radius, steps, step_size, args,
                 random_start=False, ascending=False):
        
        self.radius       = radius / 255.
        self.steps        = steps
        self.step_size    = step_size / 255.
        self.random_start = False
        self.ascending    = ascending
        self.delta        =  torch.zeros([fin_num, 3, 32, 32]).uniform_(-self.radius, self.radius).to(args.device)

    def perturb(self, model, criterion, x, y):
        if self.steps==0 or self.radius==0:
            return torch.tensor(0)
        
        ''' initialize noise '''
        delta = self.delta.data

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()

        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()

        for step in range(self.steps):
            delta.grad = None
            
            adv_x = (x + delta).clamp(0., 1)
            _y = model(adv_x)
            lo = criterion(_y, y) 
            print(lo.item())
            lo.backward()
            if torch.isnan(lo).any():
                print('has nan!')
            with torch.no_grad():
                grad = delta.grad.data
                if not self.ascending: grad.mul_(-1)
                delta.add_(torch.sign(grad), alpha=self.step_size)
                delta.clamp_(-self.radius, self.radius)
        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True
        
        self.delta = delta.data
        return delta.data



def train_mimic(args):
    model = utils.get_pretrained_model(args)
    model.to(args.device)
    model.train()
    
    data_file = torch.load(args.data_path + '/' + args.dataset + '/allocated_data/data_log.pth')
    
    adv_train_index = torch.randperm(len(data_file["X_train"]))[0:args.adv_train_num]
    train_ldr = DataLoader(TensorDataset(data_file["X_train"][adv_train_index], data_file["y_train"][adv_train_index]), 
                           batch_size = 64, shuffle=True)
    test_ldr = DataLoader(TensorDataset(data_file["X_remain"], data_file["y_remain"]), 
                           batch_size = 128, shuffle=False)
    
    

    maximizing = utils.PGDAttacker(
                radius       = args.adv_radius,
                steps        = args.adv_steps,
                step_size    = args.adv_step_size,
                random_start = True,
                norm_type    = 'l-infty',
                ascending    = True)
    
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=args.mimic_lr,
                            momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, 
                                                     milestones=[10, 20], 
                                                     gamma=0.5)
    
    save_dir = './Source_models/' + args.dataset + '/' + args.arch + '/Mimic_models/'
    os.makedirs(save_dir, exist_ok=True)
    logger = utils.get_logger(save_dir + 'log.txt')
    logger.info('start training!')
    
    
    for epoch in range(args.num_mimic):
        loss_total = 0
        # calculate test acc
        test_acc_meter = utils.test_acc_dataldr(model, args.device, test_ldr)
        
        for batch_idx, (x, y) in enumerate(train_ldr):
            model.train()
            x, y = x.to(args.device), y.to(args.device)
            select_index = torch.randperm(len(x))[0:8]
            adv_x = maximizing.perturb(model, criterion, x[select_index], y[select_index]) 
            
            x = torch.concat([x,adv_x])
            y = torch.concat([y,y[select_index]])
            y_pred = model(x)
            train_loss = criterion(y_pred, y)
            
            loss_total += train_loss.cpu().data.item()
            
            optim.zero_grad()
            train_loss.backward()
            optim.step()
        
        scheduler.step()
        torch.save(model.state_dict(), save_dir+'epoch_'+str(epoch+1)+'.pth')
        logger.info('''Epoch:[{}/{}]\t loss={:.5f}\t test_acc={:.4f}\t'''.format(epoch , args.num_mimic, loss_total, test_acc_meter))


def generate_UF_samples(args):
    data_file = torch.load(args.data_path + '/' + args.dataset + '/allocated_data/data_log.pth')
    x_record = data_file["X_fin"].to(args.device)
    y_record = data_file["y_fin"].to(args.device)
    
    uf_generator = utils.PGD_uf(
                        fin_num          = args.num_UF,
                        radius           = args.UF_radius,
                        steps            = args.UF_steps,
                        step_size        = args.UF_step_size,
                        args = args)
    criterion = torch.nn.CrossEntropyLoss()
    
    
    adv_model_dir = './Source_models/' + args.dataset + '/' + args.arch + '/Mimic_models/'
    for step in range(args.UF_iter):
        current_index = np.random.randint(args.num_mimic) + 1
        if current_index == args.num_mimic:
            model = utils.get_pretrained_model(args)
        else:
            current_model = adv_model_dir + 'epoch_' + str(current_index) + '.pth'
            model = utils.get_pretrained_model(args, specify=current_model)
        model = model.to(args.device)
        model.eval()
        x, y = x_record, y_record
        
        delta = uf_generator.perturb(model, criterion, x, y)
        if step == 100:
            uf_generator.step_size = uf_generator.step_size * 0.9
        
    adv_x = x + delta
    adv_x.clamp_(0, 1)
    save_UF = {}
    save_UF['X'] = adv_x.data.cpu()
    save_UF['y'] = y_record.data.cpu()

    fin_dir = './fingerprints/' + args.dataset + '/' + args.arch + '/UF_samples.pth'
    os.makedirs(fin_dir, exist_ok=True)
    torch.save(save_UF, fin_dir)





















