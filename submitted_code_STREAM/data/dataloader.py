import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random, copy
import argparse
import numpy as np
from torchvision import transforms
from .dataset import *

imagenet_transforms = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
])

cifar100_transfoms = transforms.Compose([transforms.Resize([32, 32]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ])

input_size_dict = {'Tiny-imagenet': 32,
                   'SVHN': 32,
                   'CIFAR100': 32,
                   'CUB200': 224,
                   'AWA2': 224,
                   'Stanford_dogs':224,
}


class VisionDataset(object):
    def __init__(self, args):
        self.args = args
        if args.dataset =='CIFAR100':
            self.supervised_trainloader = get_loader(self.args, indices=None,  transforms=cifar100_transfoms, train=True)
            self.supervised_testloader = get_loader(self.args, indices=None,  transforms=cifar100_transfoms, train=False)
        if args.dataset == 'Tiny-imagenet':
            self.supervised_trainloader = get_loader(self.args, indices=None, transforms=imagenet_transforms, train=True)
            self.supervised_testloader = get_loader(self.args, indices=None, transforms=imagenet_transforms, train=False)
        # if args.dataset == 'SVHN':
        #     self.train_class_labels_dict, self.test_class_labels_dict = classwise_split(
        #         targets=self.supervised_trainloader.dataset.labels), classwise_split(
        #         targets=self.supervised_testloader.dataset.labels)

        self.train_class_labels_dict, self.test_class_labels_dict = classwise_split(
            targets=self.supervised_trainloader.dataset.targets), classwise_split(
            targets=self.supervised_testloader.dataset.targets)

        self.num_classes = len(self.train_class_labels_dict)
        self.num_tasks = args.n_tasks
        cl_class_list = list(range(self.num_classes))
        # random.shuffle(cl_class_list)


        self.data_loaders = {'sequential':{}}
        # self.test_task_loaders = []
        self.n_input = input_size_dict[args.dataset]

        # task loader
        for i in range(self.num_tasks):
            self.data_loaders['sequential'][i+1] = {}
            trainidx = []
            testidx = []
            selected_cls  = cl_class_list[self.args.n_cls_per_task*i: self.args.n_cls_per_task*(i+1)]
            trainidx += [self.train_class_labels_dict[k] for k in selected_cls]
            testidx += [self.test_class_labels_dict[k] for k in selected_cls]

            noise_target_transform = ProcessTargets(self.args, i)
            train_loader = get_loader(args, indices=np.concatenate(trainidx), transforms=cifar100_transfoms, train=True, shuffle=True,\
                                      target_transforms=noise_target_transform)
            test_loader = get_loader(args, indices=np.concatenate(testidx), transforms=cifar100_transfoms, train=False)
            self.data_loaders['sequential'][i+1]['train'] = train_loader
            self.data_loaders['sequential'][i+1]['val'] = test_loader




def get_loader(args, indices, transforms, train, shuffle=True, target_transforms=None):
    sampler = None
    if indices is not None: sampler = SubsetRandomSampler(indices) if (shuffle and train) else SubsetSequentialSampler(
        indices)

    if args.dataset == 'CUB200':
        split = 'train' if train else 'test'
        dataset = CUB200(data_dir=args.data_path, split=split, transform=transforms,
                         target_transform=target_transforms)
        return DataLoader(dataset, sampler=sampler, num_workers=0, batch_size=args.batch_size)
    elif args.dataset == 'AWA2':
        split = 'train' if train else 'test'
        dataset = AWA2(data_dir=args.data_path, split=split, transform=transforms,
                       target_transform=target_transforms)
        return DataLoader(dataset, sampler=sampler, num_workers=0, batch_size=args.batch_size)
    elif args.dataset == 'CIFAR100':
        split = True if train else False
        dataset = torchvision.datasets.CIFAR100(download=True, root=args.data_path, train=split, transform=transforms, target_transform=target_transforms)
        return DataLoader(dataset, sampler=sampler, num_workers=0, batch_size=args.batch_size)
    elif args.dataset == 'Tiny-imagenet':
        split = 'train' if train else 'val'
        dataset = torchvision.datasets.ImageFolder(f'./data/Tiny-Imagenet/{split}/',  transform=transforms, target_transform=target_transforms)
        # dataset = torchvision.datasets.Tiny(download=False, root=args.data_path, train=split, transform=transforms,
        #                                         target_transform=target_transforms)
        return DataLoader(dataset, sampler=sampler, num_workers=0, batch_size=args.batch_size)
    else:
        raise 'error'


def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class ProcessTargets(object):
    """
    add noise to labels
    """
    def __init__(self, args, task):
        self.noise = args.label_noise
        self.noise_rate = args.noise_rate
        self.n_cls_per_task = args.n_cls_per_task
        self.task = task

    def __call__(self, target):
        if self.noise:
            convert = np.random.rand(1)[0] < self.noise_rate
            if convert:
                cls = list(range(self.task*self.n_cls_per_task, (self.task+1)*self.n_cls_per_task))
                cls.remove(target)
                target = np.long(np.random.choice(cls, 1)[0])
        return target


def get_loaders(self, mixture, ncla, name, imb_idx=None):
    loaders = {'sequential': {},  'multitask':  {}, 'subset': {}, 'coreset': {}, 'full-multitask': {}}
    print('loading coreset placeholder MixtureDataset')
    for task in range(1, self.opt.n_tasks+1):
        loaders['sequential'][task], loaders['coreset'][task], loaders['subset'][task] = {}, {}, {}
        print("loading {} for task {}".format(name[task-1], task))
        seq_loader_train , seq_loader_val = get_mixture_loader(task, self.opt.batch_size, mixture[task-1], ncla, full_train=False, imb_idx=imb_idx)
        loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
    return loaders