from PIL import Image
import numpy as np
import torch
import torchvision
import os
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_data(dataset, data_root):

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                              ])  
       

        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform
                                               )


        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=True,
                                                transform=transform
                                                )
        all_dataset = torch.utils.data.dataset.ConcatDataset([train_set, test_set])
    
    if dataset == 'cifar100':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(15),
                                        transforms.ToTensor()
                                              ])  

        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform
                                               )


        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=True,
                                                transform=transform
                                                )
        all_dataset = torch.utils.data.dataset.ConcatDataset([train_set, test_set])
                
    if dataset == "ImageNet":
        test_set = torchvision.datasets.ImageNet(data_root,
                                                split = 'val',
                                                transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                              transforms.RandomRotation(15),
                                                                              transforms.ToTensor()
                                                                              ])
                                                )
        return test_set

    return all_dataset


from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import sys


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt






class DatasetSplit(Dataset):
    def __init__(self, dataset, num_data):
        self.dataset = dataset
        idxs = np.arange(len(dataset))
        self.idxs = np.random.choice(idxs,num_data,replace=False)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def allocate_imagenet(args):
    
    transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor()
                                ])
    data_dir = './Data/tiny-imagenet-200'
    dataset_train = TinyImageNet(data_dir, train=True, transform=transform)

    X = []
    y = []
    for data in dataset_train:
        X.append(data[0].unsqueeze(0))
        y.append(data[1])
    X = torch.cat(X, axis=0)
    y = torch.tensor(y)
    
    # shuffle the dataset
    idx = torch.randperm(len(dataset_train))
    X = X[idx]
    y = y[idx]

    
    # allocate data
    data_log = {}
    data_log["X_train"] = X[0:args.num_train]
    data_log["y_train"] = y[0:args.num_train]
    data_log["X_fin"] = X[args.num_train:args.num_train+args.fin_num]
    data_log["y_fin"] = y[args.num_train:args.num_train+args.fin_num]
    data_log["X_remain"] = X[args.num_train+args.fin_num:]
    data_log["y_remain"] = y[args.num_train+args.fin_num:]
    
    # save data
    data_dir = args.data_path + '/' + args.dataset + '/allocated_data'
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data_log, data_dir + '/data_log.pth')


def allocate_data(args):
    dataset = get_data(args.dataset, data_root="./Data")
    list_loader = list(dataset)
    
    if len(list_loader) < args.num_train:
        raise Exception("Data used for training and attack is in excess of total data")

    X = []
    y = []
    for data in list_loader:
        X.append(data[0].unsqueeze(0))
        y.append(data[1])
    X = torch.cat(X, axis=0)
    y = torch.tensor(y)
    
    # shuffle the dataset
    idx = torch.randperm(len(list_loader))
    X = X[idx]
    y = y[idx]

    
    # allocate data
    data_log = {}
    data_log["X_train"] = X[0:args.num_train]
    data_log["y_train"] = y[0:args.num_train]
    data_log["X_fin"] = X[args.num_train:args.num_train+args.fin_num]
    data_log["y_fin"] = y[args.num_train:args.num_train+args.fin_num]
    data_log["X_remain"] = X[args.num_train+args.fin_num:]
    data_log["y_remain"] = y[args.num_train+args.fin_num:]
    
    # save data
    data_dir = args.data_path + '/' + args.dataset + '/allocated_data'
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data_log, data_dir + '/data_log.pth')



class ElementWiseTransform():
    def __init__(self, trans=None):
        self.trans = trans

    def __call__(self, x):
        if self.trans is None: return x
        return torch.cat( [self.trans( xx.view(1, *xx.shape) ) for xx in x] )


class IndexedTensorDataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        ''' transform HWC pic to CWH pic '''
        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1)
        return x, y, idx

    def __len__(self):
        return len(self.x)


class Dataset():
    def __init__(self, x, y, transform=None, fitr=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.fitr = fitr

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        ''' low pass filtering '''
        if self.fitr is not None:
            x = self.fitr(x)

        ''' data augmentation '''
        if self.transform is not None:
            x = self.transform( Image.fromarray(x) )

        return x, y

    def __len__(self):
        return len(self.x)


class IndexedDataset():
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.ii = np.array( range(len(x)), dtype=np.int64 )
        self.transform = transform

    def __getitem__(self, idx):
        x, y, ii = Image.fromarray(self.x[idx]), self.y[idx], self.ii[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, ii

    def __len__(self):
        return len(self.x)


def datasetCIFAR10(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)

def datasetCIFAR100(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)

def datasetTinyImageNet(root='./path', train=True, transform=None):
    if train: root = os.path.join(root, 'tiny-imagenet_train.pkl')
    else: root = os.path.join(root, 'tiny-imagenet_val.pkl')
    with open(root, 'rb') as f:
        dat = pickle.load(f)
    return Dataset(dat['data'], dat['targets'], transform)



class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=4):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples
