import torch, torchvision

import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from torchvision import transforms

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
        self.net = net
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.n_logits = args.n_logits
        self.opt = optim.Adam(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(size=(32, 32))
        ])
        # allocate episodic memory for inputs and its logit outputs
        self.memory_data = torch.zeros(
            n_tasks, self.n_memories, 3, n_inputs, n_inputs, dtype=torch.float)
        self.memory_logits = torch.zeros(
            n_tasks, self.n_memories, self.n_logits, dtype=torch.float)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_logits = self.memory_logits.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = self.args.n_classes

        self.device = 'cuda' if args.cuda == True else 'cpu'

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
            offsets = [sum(self.args.n_classes[:c]) for c in range(1, len(self.args.n_classes) + 1)]
            offset1 = int(offsets[t - 1])
            offset2 = int(offsets[t])
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < offsets[-1]:
                output[:, offset2:offsets[-1]].data.fill_(-10e10)
            return output


    def observe(self, x, t, y, epoch):
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # compute memory loss
        loss_v = 0
        self.zero_grad()
        if len(self.observed_tasks) > 1:
            sampler_per_task = self.n_memories // (len(self.observed_tasks))
            m_x, m_l, m_y = [], [], []
            for tt in range(len(self.observed_tasks) - 1):
                # sampling the memory data
                past_task = self.observed_tasks[tt]
                index = torch.randperm(len(self.memory_data[past_task]))[:sampler_per_task]
                m_x.append(self.memory_data[past_task][index])
                m_l.append(self.memory_logits[past_task][index])

            m_x, m_l = torch.cat(m_x), torch.cat(m_l)
            # shuffle
            index = torch.randperm(len(m_x))
            m_x, m_l = m_x[index], m_l[index]
            m_x = self.transforms(m_x)
            output = self.net(m_x)
            # kl loss
            loss_v = self.args.alpha * F.mse_loss(output, m_l)

        # compute the current loss
        offset1, offset2 = compute_offsets(self.args.n_classes, t)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        # total loss
        loss += loss_v
        loss.backward()
        self.opt.step()

        # Update the buffer by storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t-1, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        self.memory_logits[t-1, self.mem_cnt: endcnt].copy_(
            self.net(x).data[: effbsz])
        self.mem_cnt += effbsz
        # if full, set the pointer to the starter, will replace the earliest samples with new samples
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        return loss.item()