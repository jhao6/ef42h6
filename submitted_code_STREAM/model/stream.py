import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision
from torchvision import transforms
from .common import MLP, ResNet18


def compute_offsets(task, n_classes, t):
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
        self.net = net
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.Adam(self.parameters(), args.lr) \
            if args.dataset != 'mixture' else optim.SGD(self.parameters(), args.lr)
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(size=(32, 32))
        ])
        self.n_memories = args.n_memories
        self.gpu = args.cuda
        self.current_loss = 0
        self.batch_idx = 0
        # allocate episodic memory
        self.memory_data = torch.zeros(
            n_tasks, self.n_memories, 3, n_inputs, n_inputs, dtype=torch.float)
        self.memory_labs = torch.zeros(n_tasks, self.n_memories, dtype=torch.long)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = 0
        self.mem_cnt = 0
        self.loss_v = 0
        self.past_loss = 0
        self.beta = args.beta
        self.cur_threshold = args.threshold

        self.nc_per_task = self.args.n_classes

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
        if t != self.old_task:
            self.cur_threshold = self.args.threshold/t if self.args.thre_decay else self.args.threshold

            self.observed_tasks.append(t)
            if t>1:
                self.past_loss +=  self.current_loss/self.batch_idx
                self.past_loss_avg =  self.past_loss/self.old_task
            self.old_task = t
            self.current_loss = 0
            self.batch_idx = 0

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t-1, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t-1, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t-1, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        self.zero_grad()
        # memory
        if len(self.observed_tasks) > 1:
            sampler_per_task = self.n_memories // (len(self.observed_tasks))
            m_x, m_y = [], []
            for tt in range(len(self.observed_tasks) - 1):
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                index = torch.randperm(len(self.memory_data[past_task]))[:sampler_per_task]
                m_x.append(self.memory_data[past_task][index])
                m_y.append(self.memory_labs[past_task][index])
            m_x, m_y = torch.cat(m_x), torch.cat(m_y)
            # shuffle
            index = torch.randperm(len(m_x))
            m_x, m_y = m_x[index], m_y[index]
            m_x = self.transforms(m_x)
            output = self.net(m_x)
            loss_m = self.ce(output, m_y)
            self.loss_v = self.beta * self.loss_v + (1 - self.beta) * loss_m.detach()

        # now compute the grad on the current minibatch
        offset1, offset2 = compute_offsets(t, self.args.n_classes, t)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        if epoch == self.args.epochs-1:
            self.current_loss += loss.item()
            self.batch_idx += 1

        if t>1 and self.loss_v - self.past_loss_avg > self.cur_threshold:

            loss_m.backward()
        else:
            loss.backward()

        self.opt.step()
        return loss.item()

