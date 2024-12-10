

import torch
from .common import MLP, ResNet18
def compute_offsets( n_classes, t):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    offsets = [sum(n_classes[:c]) for c in range(1, len(n_classes) + 1)]
    offset1 = int(offsets[t - 1])
    offset2 = int(offsets[t])
    return offset1, offset2

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 net,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.reg = args.memory_strength
        self.is_cifar_isic = True
        # setup network
        self.net = net
        self.args = args
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 1
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None


        self.n_outputs = n_outputs
        self.n_memories = args.n_memories



    def forward(self, x, t):
        output = self.net(x)
        # make sure we predict classes within the current task
        if isinstance(self.args.n_classes, int):
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
        self.net.train()

        # next task?
        if t != self.current_task:
            self.net.zero_grad()
            offset1, offset2 = compute_offsets( self.args.n_classes, t-1)
            self.bce((self.net(self.memx)[:, offset1: offset2]),
                     self.memy - offset1).backward()
            self.fisher[self.current_task-1] = []
            self.optpar[self.current_task-1] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task-1].append(pd)
                self.fisher[self.current_task-1].append(pg)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.net.zero_grad()
        if self.is_cifar_isic:
            offset1, offset2 = compute_offsets(self.args.n_classes, t)
            loss = self.bce((self.net(x)[:, offset1: offset2]),
                            y - offset1)
        else:
            loss = self.bce(self(x, t), y)
        if t > 1:
            for tt in range(t-1):
                for i, p in enumerate(self.net.parameters()):
                    l = self.reg * self.fisher[tt][i]
                    l = l * (p - self.optpar[tt][i]).pow(2)
                    loss += l.sum()
        loss.backward()
        self.opt.step()
        return loss.item()
