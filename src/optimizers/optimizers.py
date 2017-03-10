from torch.optim import Optimizer
from torch.optim.optimizer import required


class MySGD(Optimizer):
    """Simple SGD"""

    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad.data is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss
