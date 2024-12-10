
import torch
from .utils import Reservoir


def compute_offsets(n_classes, t):
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
        self.is_cifar_isic = True
        # setup network
        self.net = net
        self.args = args
        # setup optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()


        self.nc_per_task = self.args.n_classes
        self.n_outputs = n_outputs
        self.memory = Reservoir(n_tasks * args.n_memories, (3, n_inputs, n_inputs))



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
        self.train()
        self.zero_grad()
        self.memory.update(x, y, t)
        m_x, m_y, m_t = self.memory.sample(sample_size=x.shape[0])
        loss = self.bce(self.net(m_x), m_y)
        loss.backward()
        self.opt.step()
        return loss.item()
