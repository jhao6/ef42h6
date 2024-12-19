import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from metrics.metrics import confusion_matrix
from data.dataloader import VisionDataset
from model.common import ResNet18
from data.dataset import MixtureDataset as MD
import warnings
warnings.filterwarnings('ignore')

def eval_tasks(model, t, args):
    model.eval()
    result = []
    for t in range(1, args.n_tasks+1):
        rt = 0
        data_size = 0
        for batch_dix, (xb, yb) in enumerate(dataset['sequential'][t]['val']):
            with torch.no_grad():
                if args.cuda:
                    xb = xb.cuda()
                    out = model(xb, t)
                    _, pb = torch.max(out.data.cpu(), 1, keepdim=False)
                    rt += (pb == yb).float().sum()
                    data_size += yb.size(0)
        result.append(rt / data_size)

    return torch.tensor(result)


def life_experience(model, dataset, args):
    result_a = []
    result_t = []

    time_start = time.time()
    loss = []
    avg_result_a = []
    for t in range(1, args.n_tasks+1):
        print(f'Training on task {t}')
        for epoch in range(args.epochs):
            loss_t = []
            for batch_dix, (x, y)  in enumerate(dataset['sequential'][t]['train']):
                x = x.cuda()
                y = y.cuda()
                model.train()
                temp_loss = model.observe(x, t, y, epoch)
                loss_t.append(temp_loss)
                print(f'Epoch [{epoch}/{args.epochs}]：loss:{temp_loss:.4f}')
            loss.append(torch.mean(torch.tensor(loss_t)))

        res_acc = eval_tasks(model, t, args)
        result_a.append(res_acc)
        avg_result_a.append(res_acc[:t].mean().round(decimals=4).item())
        print(f'Task [{t}/{args.n_tasks}] Avg ACC: {avg_result_a}')
        result_t.append(t)
    time_end = time.time()
    time_spent = time_end - time_start
    plt.figure()
    plt.plot(range(len(avg_result_a)), avg_result_a)
    plt.xlabel('Epochs')
    plt.title('Training loss vs. Epochs')
    plt.xlim()
    plt.grid()

    fig_save_path = os.path.join(args.save_path, 'figures')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    fig_save_path = os.path.join(args.save_path, 'figures')
    plt.savefig(f'{fig_save_path}/test_acc_{args.model}_{args.seed}.eps')

    return avg_result_a, torch.vstack(result_a), time_spent/3600


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='stream',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=64,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0.5, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs per task')
    parser.add_argument('--mem_batch_size', type=int, default=64,
                        help='number of memories per task')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='SGD learning rate')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='momentum for loss')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='memory loss coefficient')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='threshold for updating constrain')
    parser.add_argument('--thre_decay', type=str, default='no',
                        help='threshold decay')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='yes',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--n_classes', type=int, default=[0, 10, 10, 10, 43, 10],
                        help='the number of classes in each task')
    parser.add_argument('--n_logits', type=int, default=100,
                        help='the dimension of logits')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='data',
                        help='path where data is located')
    parser.add_argument('--dataset', default='CIFAR100',
                        help='data file [mixture, CIFAR100, Tiny-imagenet]')
    parser.add_argument('--n_tasks', default=20,
                        help='task numbers')
    parser.add_argument('--n_cls_per_task', default=5,
                        help='class number of each task')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--label_noise', type=str, default='no',
                        help='noise of label')
    parser.add_argument('--noise_rate', type=float, default=0.3,
                        help='noise rate for label')
    parser.add_argument('--apply_augmentation', type=bool, default=False,
                        help='apply data augmentation to buffer data or not')
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False
    args.label_noise = True if args.label_noise == 'yes' else False
    args.thre_decay = True if args.thre_decay == 'yes' else False
    args.data_file  = ''
    # unique identifier
    uid = uuid.uuid4().hex

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    if args.dataset in ['CIFAR100']:
        args.n_tasks = 20
        args.epochs = 10
        args.n_cls_per_task = 5
        args.n_memories = 256
        args.lr = 0.001
        args.batch_size = 128
        args.n_classes = [0]+ [args.n_cls_per_task]*args.n_tasks
        args.data_path = "data/CIFAR100"
        datasets = VisionDataset(args)
        n_outputs = datasets.num_classes
        args.n_logits = n_outputs
        dataset = datasets.data_loaders
        net = ResNet18(nclasses=n_outputs, args=args)
        n_inputs = 32
        if args.label_noise:
            args.save_path = os.path.join(args.save_path, f'{args.dataset}_task{args.n_tasks}_noise{args.noise_rate}')
        else:
            args.save_path = os.path.join(args.save_path, f'{args.dataset}_task{args.n_tasks}')

    if args.dataset in ['Tiny-imagenet']:
        args.n_tasks = 20
        args.lr= 0.001
        args.epochs = 15
        args.n_cls_per_task = int(200/args.n_tasks)
        args.n_memories = 256
        args.n_classes = [0]+ [args.n_cls_per_task]*args.n_tasks
        args.data_path = "data/Tiny-Imagenet"
        datasets = VisionDataset(args)
        n_outputs = datasets.num_classes
        args.n_logits = n_outputs
        dataset = datasets.data_loaders
        net = ResNet18(nclasses=n_outputs, args=args)
        n_inputs = 32
        if args.label_noise:
            args.save_path = os.path.join(args.save_path, f'{args.dataset}_task{args.n_tasks}_noise{args.noise_rate}')
        else:
            args.save_path = os.path.join(args.save_path, f'{args.dataset}_task{args.n_tasks}')

    if args.dataset in ['mixture']:
        n_outputs=83
        net = ResNet18(nclasses=n_outputs, args=args)
        args.n_tasks=5
        if args.label_noise:
            if os.path.exists(f'data/mixture/mixture_data_noise_{args.noise_rate}.pkl'):
                dataset = torch.load(f'data/mixture/mixture_data_noise_{args.noise_rate}.pkl')
            else:
                md = MD(args, args.noise_rate, label_noise=args.label_noise)
                dataset = md.get_loader()
                torch.save(dataset, f'data/mixture/mixture_data_noise_{args.noise_rate}.pkl')
            args.save_path = os.path.join(args.save_path, f'{args.dataset}_task{args.n_tasks}_noise{args.noise_rate}')
        else:
            if os.path.exists('data/mixture/mixture_data.pkl'):
                dataset = torch.load('data/mixture/mixture_data.pkl')
            else:
                md = MD(args, args.noise_rate, label_noise=args.label_noise)
                dataset = md.get_loader()
                torch.save(dataset, 'data/mixture/mixture_data.pkl')
            args.save_path = os.path.join(args.save_path, f'{args.dataset}_task{args.n_tasks}')
        n_inputs = 32
        args.n_logits = n_outputs
        args.alpha = 0.25

    print(args)
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, args.n_tasks, net, args)
    if args.cuda:
        model.cuda()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.dataset + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)


    avg_result_a, result_a, spent_time = life_experience(model, dataset, args)
    # prepare saving path and file name
    args.save_path = os.path.join(args.save_path, 'log')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.dataset + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += f'_seed{args.seed}'
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(avg_result_a,  result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))
    print(f'test_result:{avg_result_a}')
    print(f'avg acc: {avg_result_a[-1]}, avg fgt: {stats[0]}')
    # save all results in binary file
    torch.save((avg_result_a, result_a, model.state_dict(),
                stats, one_liner, args), fname + '.pt')
