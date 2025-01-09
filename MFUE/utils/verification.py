#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import utils
from torch.utils.data import DataLoader,TensorDataset
import os


class RobustPGDAttacker():
    def __init__(self, samp_num, trans,
        radius, steps, step_size, random_start, ascending=True):
        self.samp_num     = samp_num
        self.trans        = trans

        self.radius       = radius / 255.
        self.steps        = steps
        self.step_size    = step_size / 255.
        self.random_start = random_start
        self.ascending    = ascending

    def perturb(self, model, criterion, x, y):
        ''' initialize noise '''
        delta = torch.zeros_like(x.data)
        if self.steps==0 or self.radius==0:
            return delta

        if self.random_start:
            delta.uniform_(-self.radius, self.radius)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()
        for step in range(self.steps):
            delta.grad = None

            for i in range(self.samp_num):
                adv_x = self.trans( (x + delta * 255).clamp(0., 255.) )
                _y = model(adv_x)
                lo = criterion(_y, y)
                lo.backward()

            with torch.no_grad():
                grad = delta.grad.data
                if not self.ascending: grad.mul_(-1)
                delta.add_(torch.sign(grad), alpha=self.step_size)
                delta.clamp_(-self.radius, self.radius)

        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return delta.data



class PGDDefender():
    def __init__(self, samp_num, trans,
        radius, steps, step_size, random_start=True, ascending=False):
        self.samp_num     = samp_num
        self.trans        = trans

        self.radius       = radius / 255.
        self.steps        = steps
        self.step_size    = step_size / 255.
        self.random_start = True
        self.ascending    = ascending

    def perturb(self, model, criterion, x, y):
        ''' initialize noise '''
        delta = torch.zeros_like(x.data)
        if self.steps==0 or self.radius==0:
            return delta

        if self.random_start:
            delta.uniform_(-self.radius, self.radius)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()

        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()

        for step in range(self.steps):
            delta.grad = None
            for i in range(self.samp_num):
                adv_x = self.trans((x + delta * 255).clamp(0., 255.))
                _y = model(adv_x)
                lo = criterion(_y, y) 
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

        return delta.data



class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type, ascending=True):
        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start
        self.norm_type = norm_type
        self.ascending = ascending

    def perturb(self, model, criterion, x, y):
        if self.steps==0 or self.radius==0:
            return x.clone()

        adv_x = x.clone()
        
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_x.requires_grad_()
            _y = model(adv_x)
            loss = criterion(_y, y)
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0],-1)**2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0],-1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)
                self._clip_(adv_x, x) 
        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0],-1)**2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0],-1).abs().sum(dim=1)
            norm = norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(0, 1)


def compute_acc(args, alpha, sample, label, model, perturbation):
    pb_samples = alpha*sample + (1-alpha) * perturbation  # torch.Size([100, 3, 32, 32])
    pb_samples = pb_samples.clamp(0., 1)
    pb_samples = pb_samples.to(args.device)
    pred_y = model(pb_samples).argmax(dim=1)
    acc = (pred_y==label).sum()/args.num_pb
    return pb_samples, acc

def alpha_determine(args, sample, label):
    model = utils.get_pretrained_model(args)
    model = model.to(args.device)
    model = model.eval()
    threshold_low = 0.4
    threshold_high = 0.6
    acc = 1.0
    torch.randn_like(sample.repeat(args.num_pb,1,1,1))
    perturbation = torch.randn_like(sample.repeat(args.num_pb,1,1,1))
    alpha_h = 1.0
    alpha_l = 0.0
    k = 0
    while acc > threshold_high or acc < threshold_low:
        k += 1
        if k > 8:
            return None, None
        alpha_m = (alpha_h + alpha_l) / 2
        pb_samples, acc = compute_acc(args, alpha_m, sample, label, model, perturbation)
        if acc < threshold_low:
            alpha_l = alpha_m   
        elif acc > threshold_high:
            alpha_h = alpha_m
        else:
            return pb_samples, -torch.log(acc)

    


def MC_estimate(args):
    spc_model = utils.get_pretrained_model(args, specify=args.spc_path)
    spc_model = spc_model.to(args.device)
    spc_model.eval()
    fin_dir = './fingerprints/' + args.dataset + '/' + args.arch + '/UF_samples.pth'
    UF_samples = torch.load(fin_dir)['X']
    UF_labels = torch.load(fin_dir)['y']
    count_similar = 0
    count_num = 0
    for sample,label in zip(UF_samples,UF_labels):
        pb_samples, source_loss = alpha_determine(args, sample, label)
        if pb_samples == None:
            continue
        count_num += 1
        pb_samples = pb_samples.to(args.device)
        pred_y = spc_model(pb_samples).argmax(dim=1)
        acc = (pred_y==label).sum()/args.num_pb
        spc_loss = -torch.log(acc)
        tol = args.tolerance_factor
        if (spc_loss < (1+tol)*source_loss) and (spc_loss > (1-tol)*source_loss):
            count_similar+=1
    return count_similar/count_num
    
