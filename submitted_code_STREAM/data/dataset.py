import os

import cv2
import torch
from PIL import Image
import random
import numpy as np
from data.mixture import get

def merge_classes(target, n_class, n_tasks):
    assert n_class >= 2 * n_tasks
    nc_per_task = n_class // (2 * n_tasks)
    target = list(np.array(target) // nc_per_task)
    return target

class CUB200(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform, target_transform, n_tasks=20):
        self.data_dir = data_dir
        self.split = split # train or test
        self.transform = transform
        self.target_transform = target_transform
        CUB_TRAIN_LIST = 'dataset_lists/CUB_train_list.txt'
        CUB_TEST_LIST = 'dataset_lists/CUB_test_list.txt'
        if split == 'train':
            file_name = CUB_TRAIN_LIST
        elif split == 'test':
            file_name = CUB_TEST_LIST
        else:
            raise 'error'
        self.img_paths = []
        self.targets = []
        with open(file_name) as f:
            for line in f:
                img_name, img_label = line.split()
                img_path = data_dir.rstrip('\/') + '/' + img_name
                self.img_paths.append(img_path)
                self.targets.append(int(img_label))

        self.targets = merge_classes(self.targets, n_class=200, n_tasks=n_tasks)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class AWA2(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform, target_transform):
        self.data_dir = data_dir
        self.split = split # train or test
        self.transform = transform
        self.target_transform = target_transform
        AWA_TRAIN_LIST = 'dataset_lists/AWA_train_list.txt'
        AWA_TEST_LIST = 'dataset_lists/AWA_test_list.txt'
        if split == 'train':
            file_name = AWA_TRAIN_LIST
        elif split == 'test':
            file_name = AWA_TEST_LIST
        else:
            raise 'error'
        self.img_paths = []
        self.targets = []
        with open(file_name) as f:
            for line in f:
                img_name, img_label = line.split()
                img_path = data_dir.rstrip('\/') + '/' + img_name
                self.img_paths.append(img_path)
                self.targets.append(int(img_label))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_mixture_loader(task_id, batch_size, dataset, ncla, shuffle=True, full_train=True, train_size=1000, imb_idx=None, device='cpu'):
    start_class = sum(ncla[:task_id-1]) if task_id > 1 else 0
    end_class = sum(ncla[:task_id])

    dataset['train']['y'] += start_class
    dataset['test']['y'] += start_class

    trains, evals = [], []
    total_train = len(dataset['train']['y'])
    if shuffle:
        shuf = torch.randperm(total_train)
        dataset['train']['x'] = dataset['train']['x'][shuf]
        dataset['train']['y'] = dataset['train']['y'][shuf]

    # imbalanced
    if isinstance(imb_idx, list) and ncla[task_id-1] == 10:
        idx_per_class = []
        for class_number in range(start_class, end_class):
            cid_index = np.asarray([i for i, c in enumerate(dataset['train']['y']) if c == class_number])
            idx_per_class.append(torch.from_numpy(cid_index[:imb_idx[class_number]]))

        reduced = torch.cat(idx_per_class)
        _shuffle = torch.randperm(len(reduced))

        dataset['train']['x'] = dataset['train']['x'][reduced[_shuffle].long()]
        dataset['train']['y'] = dataset['train']['y'][reduced[_shuffle].long()]

    iaa = []
    for class_number in range(start_class, end_class):
        cid_index = np.asarray([i for i, c in enumerate(dataset['train']['y']) if c == class_number])
        iaa.append(len(cid_index))
    print(iaa)

    if full_train:
        iteration = round(total_train / batch_size)
    else:
        t_size = min(len(dataset['train']['x']), train_size)
        iteration = round(t_size / batch_size)
    index = 0
    for i in range(iteration):
        offset = min(batch_size, total_train-index)
        data = dataset['train']['x'][index:index+offset].to(device)
        target = dataset['train']['y'][index:index+offset].to(device)
        trains.append([data, target])
        index += batch_size

    total_test = len(dataset['test']['y'])
    iteration = round(total_test / batch_size)
    index = 0
    for i in range(iteration):
        offset = min(batch_size, total_test-index)
        data = dataset['test']['x'][index:index+offset].to(device)
        target = dataset['test']['y'][index:index+offset].to(device)
        evals.append([data, target])
        index += batch_size

    return trains, evals

class MixtureDataset:
    def __init__(self, opt, noise_rate, valid=False, is_imbalanced=False, label_noise=False):
        self.is_imb = is_imbalanced
        self.label_noise = label_noise
        self.noise_rate = noise_rate
        self.opt = opt
        if valid:
            self.seprate_ratio = (0.7, 0.2, 0.1) # train, test, valid
        else:
            self.seprate_ratio = (0.7, 0.3)
        self.mixture_dir = './data/mixture/'
        self.mixture_filename = 'mixture2.npy'
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }

    def get_imbalanced_idx(self, n_class=40, img_max=int(1000/10)):
        class_shuffle = [31,  9, 11, 39, 32, 34, 13,  5, 20, 29,  1, 36, 24, 14, 25, 26,  7, 12,\
                            23, 18, 10, 38,  8, 17, 27,  3, 35,  2, 28,  0, 21,  6, 37, 15, 33, 22, 19, 16, 30,  4]

        len_per_class = []
        imbalanced_idx = []
        imb_factor = 1/10.
        for class_number in range(n_class):
            num_sample = int(img_max * (imb_factor**(class_number/(n_class - 1))))
            len_per_class.append(num_sample)
        out =  np.array(len_per_class)[class_shuffle].tolist()
        oqq = out[:30] + [0 for _ in range(43)] + out[30:]
        return oqq

    def get_loader(self):
        return self.generate_data()

    def generate_data(self):
        saved_mixture_filepath = os.path.join(self.mixture_dir, self.mixture_filename)
        if os.path.exists(saved_mixture_filepath):
            print('loading mixture data: {}'.format(saved_mixture_filepath))
            mixture = np.load(saved_mixture_filepath, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            mixture = get(base_dir=self.mixture_dir, pc_valid=0.0, fixed_order=True, label_noise=self.label_noise, \
                          noise_rate=self.noise_rate)
        ncla = [mixture[0][tid]['ncla'] for tid in range(5)]
        name = [mixture[0][tid]['name'] for tid in range(5)]

        if self.is_imb:
            imb_idx = self.get_imbalanced_idx()
            loader = self.get_loaders(mixture[0], ncla, name, imb_idx)
        else:
            loader = self.get_loaders(mixture[0], ncla, name)
        torch.save(loader, saved_mixture_filepath)
        return loader

    def get_noise_loaders(self):
        pass

    def get_loaders(self, mixture, ncla, name, imb_idx=None):
        loaders = {'sequential': {},  'multitask':  {}, 'subset': {}, 'coreset': {}, 'full-multitask': {}}
        print('loading coreset placeholder MixtureDataset')
        for task in range(1, self.opt.n_tasks+1):
            loaders['sequential'][task], loaders['coreset'][task], loaders['subset'][task] = {}, {}, {}
            print("loading {} for task {}".format(name[task-1], task))
            seq_loader_train , seq_loader_val = get_mixture_loader(task, self.opt.batch_size, mixture[task-1], ncla, full_train=False, imb_idx=imb_idx)
            loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
        return loaders