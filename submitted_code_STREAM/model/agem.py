
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

from .common import MLP, ResNet18
import random
# Auxiliary functions useful for GEM's inner optimization.

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

        self.opt = optim.Adam(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.zeros(
            n_tasks, self.n_memories, 3, n_inputs, n_inputs, dtype=torch.float)
        self.memory_labs = torch.zeros(n_tasks, self.n_memories, dtype=torch.long)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = self.args.n_classes
        self.reference_gradients = None

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
            self.observed_tasks.append(t)
            self.old_task = t

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
        if len(self.observed_tasks) > 1:
            self.zero_grad()
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
            output = self.net(m_x)

            ptloss = self.ce(output, m_y)
            ptloss.backward()

            self.reference_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.net.device)
                for n, p in self.net.named_parameters()]
            self.reference_gradients = torch.cat(self.reference_gradients)
            self.opt.zero_grad()

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(self.args.n_classes, t)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.net.device)
                for n, p in self.net.named_parameters()]
            current_gradients = torch.cat(current_gradients)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients,
                                          self.reference_gradients)
                grad_proj = current_gradients - \
                            self.reference_gradients * alpha2

                count = 0
                for n, p in self.net.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count:count + n_param].view_as(p))
                    count += n_param

        self.opt.step()
        return loss.item()