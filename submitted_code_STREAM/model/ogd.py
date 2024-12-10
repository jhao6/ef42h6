
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

from .common import MLP, ResNet18
import random


def compute_offsets(n_classes, t):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    offsets = [sum(n_classes[:c]) for c in range(1, len(n_classes) + 1)]
    offset1 = int(offsets[t - 1])
    offset2 = int(offsets[t])
    return offset1, offset2


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 net,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.net = net
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.opt = optim.Adam(self.parameters(), args.lr)
        self.mem_cnt = 0
        self.n_memories = args.n_memories
        self.samples_per_task = args.n_memories
        self.gpu = args.cuda

        self.base = [[[] for _ in range(self.n_memories)] for _ in range(self.n_tasks)]

        # allocate counters
        self.observed_tasks = []
        self.current_task = 1
        self.mem_cnt = 0

        self.nc_per_task = self.args.n_classes


    def grad_proj(self, grad):

        reference_gradients = torch.zeros(grad.size()).cuda()
        sum_grad = torch.zeros(grad.size()).cuda()
        for j in range(self.samples_per_task):
            for i in range(self.n_tasks):
                if self.base[i][j] != []:
                    sum_grad += self.base[i][j]
        reference_gradients += grad.dot(sum_grad)/sum_grad.dot(sum_grad) * sum_grad
        grad -= reference_gradients
        return grad

    def forward(self, x, t):
        output = self.net(x)

        if isinstance(self.nc_per_task, int):
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
            return output
        else:
            offsets = [sum(self.args.n_classes[:c]) for c in range(1,len(self.args.n_classes)+1)]
            offset1 = int(offsets[t-1])
            offset2 = int(offsets[t])
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < offsets[-1]:
                output[:, offset2:offsets[-1]].data.fill_(-10e10)
            return output

    def observe(self, x, t, y, epoch):
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
            self.samples_per_task = int(self.n_memories/t)
            # resize the memory for each task
            for i in range(self.n_tasks):
                self.base[i] = self.base[i][: self.samples_per_task]
            self.mem_cnt = 0

        # now compute the grad on the current minibatch
        self.zero_grad()
        offset1, offset2 = compute_offsets(self.args.n_classes, t)
        if epoch == self.args.epochs - 1:
            loss = self.forward(x, t)[:, offset1: offset2]
            # get the target loss value
            target_loss = loss[:, y - offset1].sum()
            # Update ring buffer storing examples from current task
            target_loss.backward()
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.net.device)
                for n, p in self.net.named_parameters()]
            cat_gradients = torch.cat(current_gradients)

            if t == 1:
                self.base[t - 1][self.mem_cnt] = cat_gradients.clone()
            else:
                new_gradient = self.grad_proj(cat_gradients)
                for g, new_g in zip(cat_gradients, new_gradient):
                    g -= new_g
                self.base[t - 1][self.mem_cnt] = cat_gradients.clone()
            self.mem_cnt += 1
            if self.mem_cnt >= self.samples_per_task:
                self.mem_cnt = 0

        self.zero_grad()
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()
        # project the current gradient to orthogonal space of base
        if len(self.observed_tasks) >= 1:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.net.device)
                for n, p in self.net.named_parameters()]
            new_gradient = self.grad_proj(torch.cat(current_gradients))
            count_param = 0
            for n, p in self.net.named_parameters():
                # print(f'{p.size()}')
                p.grad =  new_gradient[count_param: count_param+p.numel()].reshape(p.size())
                count_param += p.numel()

        self.opt.step()
        print(f'loss: {loss.item():.4f}')
        return loss.item()