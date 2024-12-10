import numpy as np
import torch
import random


def compute_offsets(task, nc_per_task):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    offset1 = task * nc_per_task
    offset2 = (task + 1) * nc_per_task
    return offset1, offset2


class Reservoir:
    """This is reservoir sampling, each sample has storage-probability 'buffer samples M / seen samples'
    """
    def __init__(self, mem_size, image_size, device='cuda'):
        self.mem_size = mem_size
        self.memory_data = torch.zeros(
            mem_size, image_size[0], image_size[1], image_size[2],
            dtype=torch.float, device=device)
        self.memory_labs = torch.zeros(mem_size, dtype=torch.long, device=device)
        self.memory_tasks = torch.zeros(mem_size, dtype=torch.long, device=device)
        self.seen_cnt = 0

    def update(self, x, y, t):
        for i in range(x.shape[0]):
            if self.seen_cnt < self.mem_size:
                self.memory_data[self.seen_cnt].copy_(x[i])
                self.memory_labs[self.seen_cnt].copy_(y[i])
                self.memory_tasks[self.seen_cnt].copy_(t)
            else:
                j = random.randrange(self.seen_cnt)
                if j < self.mem_size:
                    self.memory_data[j].copy_(x[i])
                    self.memory_labs[j].copy_(y[i])
                    self.memory_tasks[j].copy_(t)
            self.seen_cnt += 1
        return

    def sample(self, sample_size):
        perm = torch.randperm(len(self.memory_data))
        index = perm[:sample_size]
        x = self.memory_data[index]
        y = self.memory_labs[index]
        t = self.memory_tasks[index]
        return x, y, t


class RingBuffer:
    def __init__(self, n_tasks, n_memories, image_size, device='cuda'):
        self.memory_data = torch.zeros(
            n_tasks, n_memories, image_size[0], image_size[1], image_size[2],
            dtype=torch.float, device=device)
        self.memory_labs = torch.zeros(n_tasks, n_memories, dtype=torch.long, device=device)
        self.n_memories = n_memories
        self.old_task = -1
        self.mem_cnt = 0
        self.observed_tasks = []

    def update(self, x, y, t):
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

    def sample(self, sample_size):
        sampler_per_task = sample_size // (len(self.observed_tasks))
        m_x, m_y = [], []
        for tt in self.observed_tasks:
            index = torch.randperm(len(self.memory_data[tt]))[:sampler_per_task]
            m_x.append(self.memory_data[tt][index])
            m_y.append(self.memory_labs[tt][index])
        m_x, m_y = torch.cat(m_x), torch.cat(m_y)
        # shuffle
        index = torch.randperm(len(m_x))
        m_x, m_y = m_x[index], m_y[index]
        return m_x, m_y

class ClassBalanced:
    """This is reservoir sampling, each sample has storage-probability 'buffer samples M / seen samples'
    """
    def __init__(self, mem_size, image_size, device='cuda'):
        self.mem_size = mem_size
        self.memory_data = torch.zeros(
            mem_size, image_size[0], image_size[1], image_size[2],
            dtype=torch.float, device=device)
        self.memory_labs = torch.zeros(mem_size, dtype=torch.long, device=device)
        self.seen_cnt = 0

    def update(self, x, y, t):
        pos_indices = torch.where(y % 2 == 0)[0]
        neg_indices = torch.where(y % 2 == 1)[0]
        keep = torch.cat([pos_indices, neg_indices[:pos_indices.shape[0]]])
        x = x[keep]
        y = y[keep]

        for i in range(x.shape[0]):
            if self.seen_cnt < self.mem_size:
                self.memory_data[self.seen_cnt].copy_(x[i])
                self.memory_labs[self.seen_cnt].copy_(y[i])
            else:
                j = random.randrange(self.seen_cnt)
                if j < self.mem_size:
                    self.memory_data[j].copy_(x[i])
                    self.memory_labs[j].copy_(y[i])
            self.seen_cnt += 1
        return

    def sample(self, sample_size):
        perm = torch.randperm(len(self.memory_data))
        # perm = torch.randperm(min(self.seen_cnt, len(self.memory_data)))
        index = perm[:sample_size]
        x = self.memory_data[index]
        y = self.memory_labs[index]
        return x, y


class CBSR:
    """This is reservoir sampling, each sample has storage-probability 'buffer samples M / seen samples'
    """
    def __init__(self, mem_size, image_size, num_classes,device='cuda'):
        self.mem_size = mem_size
        self.memory_data = torch.zeros(
            mem_size, image_size[0], image_size[1], image_size[2],
            dtype=torch.float, device=device)
        self.memory_labs = torch.zeros(mem_size, dtype=torch.long, device=device)
        self.seen_cnt = 0
        self.num_classes = num_classes
        self.class_indexs = [[] for i in range(num_classes)]
        self.is_full = [False for i in range(num_classes)]

    def update(self, x, y, t):
        for i in range(x.shape[0]):
            if self.seen_cnt < self.mem_size:
                self.memory_data[self.seen_cnt].copy_(x[i])
                self.memory_labs[self.seen_cnt].copy_(y[i])
                self.class_indexs[y[i]].append(self.seen_cnt)
            else:
                if not self.is_full[y[i]]:
                    # find largest classes
                    largest_idx = 0
                    max_len = 0
                    for idx, arr in enumerate(self.class_indexs):
                        if len(arr) > max_len:
                            max_len = len(arr)
                            largest_idx = idx
                    self.is_full[largest_idx] = True
                    j = random.choice(self.class_indexs[largest_idx])
                    self.memory_data[j].copy_(x[i])
                    self.memory_labs[j].copy_(y[i])
                    self.class_indexs[largest_idx].remove(j)
                    self.class_indexs[y[i]].append(j)
                else:
                    j = random.choice(self.class_indexs[y[i]])
                    self.memory_data[j].copy_(x[i])
                    self.memory_labs[j].copy_(y[i])
            self.seen_cnt += 1
        return

    def sample(self, sample_size):
        perm = torch.randperm(min(self.seen_cnt, len(self.memory_data)))
        index = perm[:sample_size]
        x = self.memory_data[index]
        y = self.memory_labs[index]
        return x, y


class RM:
    def __init__(self, mem_size, image_size, num_classes, device='cuda'):
        self.mem_size = mem_size
        self.memory_data = torch.zeros(
            mem_size, image_size[0], image_size[1], image_size[2],
            dtype=torch.float, device=device)
        self.memory_labs = torch.zeros(mem_size, dtype=torch.long, device=device)
        self.seen_cnt = 0
        self.num_classes = num_classes
        self.class_indexs = [[] for i in range(num_classes)]
        self.is_full = [False for i in range(num_classes)]
        self.exposed_classes = 0

    def update(self, x, y, t):
        num_class = (t+1)*2
        # if memory is not filled
        for i in range(x.shape[0]):
            if self.seen_cnt < self.mem_size:
                self.memory_data[self.seen_cnt].copy_(x[i])
                self.memory_labs[self.seen_cnt].copy_(y[i])
                self.seen_cnt += 1

        # if memory if filled
        # mix stream data and memory data
        candidate_x = torch.cat([x, self.memory_data])
        candidate_y = torch.cat([y, self.memory_labs])
        # calucate uncertainty

        mem_per_cls = self.mem_size // num_class
        ret = []
        for i in range(num_class):
            cls_sample = torch.nonzero(candidate_y == i).squeeze()
            if cls_sample.dim()== 0:
                continue


            if len(cls_sample) <= mem_per_cls:
                ret.append(cls_sample)
            else:
                jump_idx = len(cls_sample) // mem_per_cls
                ## TODO sort by uncerntainty
                uncertain_samples = cls_sample[::jump_idx]
                ret.append(uncertain_samples[:mem_per_cls])

        index = torch.cat(ret)

        self.memory_data[:len(index)].copy_(candidate_x[index])
        self.memory_labs[:len(index)].copy_(candidate_y[index])
        return

    def sample(self, sample_size):
        perm = torch.randperm(min(self.seen_cnt, len(self.memory_data)))
        index = perm[:sample_size]
        x = self.memory_data[index]
        y = self.memory_labs[index]
        return x, y

