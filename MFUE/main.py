#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import utils
import random


def get_args():
    parser = argparse.ArgumentParser()
    
    # Training Settings for the source model
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'wrn-34-10', 'vgg11-bn',
                                 'vgg11-bn', 'vgg16-bn', 'vgg19-bn', 'densenet-121'],
                        help='Architecture of the source model')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='On which dataset to evaluate')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Supported devices')
    parser.add_argument('--random', type=float, default=42,
                        help='random seed')
    parser.add_argument('--num_train', type=int, default=30000,
                        help='Training set size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The amount of training data per iteration')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The parameter of the SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='The parameter of the SGD optimizer')
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Which optimizer to choose')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The path to the raw data')
    parser.add_argument('--model_path', type=str, default='./Source_models',
                        help='Where to resume the sorce model')
    
    # UF generation
    parser.add_argument('--adv-radius', type=float, default=16,
                        help='The UF perturbation radius')
    parser.add_argument('--adv-train-num', type=float, default=30000,
                        help='The size of the dataset used for adversarial training')
    parser.add_argument('--adv-steps', type=float, default=20,
                        help='The number of iteration steps for adversarial training')
    parser.add_argument('--adv-step-size', type=float, default=1.2,
                        help='The step size for adversarial training')
    parser.add_argument('--num-mimic', type=float, default=10,
                        help='The number of mimic models')
    parser.add_argument('--mimic-lr', type=float, default=0.001,
                        help='The learning rate of mimic models')
    parser.add_argument('--num-UF', type=float, default=100,
                        help='The number of UF samples')
    parser.add_argument('--UF-radius', type=float, default=8,
                        help='The UF perturbation radius')
    parser.add_argument('--UF-steps', type=int, default=10,
                        help='The number of iteration steps for UF generation')
    parser.add_argument('--UF-step-size', type=float, default=1.6,
                        help='The step size for UF generation')
    parser.add_argument('--UF-iter', type=float, default=2000,
                        help='The number of iterations for UF generation')
    
    # MC estimation
    parser.add_argument('--num-pb', type=int, default=100,
                        help='The number of probing samples')
    parser.add_argument('--spc-path', type=str, default='./Suspect_models/model.pth',
                        help='The path of the suspect model')
    parser.add_argument('--tolerance-factor', type=float, default=0.2,
                        help='Tolerance factor for matching rate calculation')
    
    
    return parser.parse_args()



def main(args):
    # allocate data
    if not os.path.exists(args.data_path + '/' + args.dataset + '/allocated_data/data_log.pth'):
        utils.allocate_data(args)

    # train the source model
    model = utils.get_arch(args.arch, args.dataset)
    utils.train_model(model, args)

    # generate UF samples
    utils.train_mimic(args)
    utils.generate_UF_samples(args)

    # Monte Carlo Estimation
    matching_rate = utils.MC_estimate(args)



if __name__ == '__main__':
    args = get_args()
    utils.random_seed(args.random)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)







