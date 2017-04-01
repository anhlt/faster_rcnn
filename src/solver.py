from torch import optim
from torch.utils.data.dataloader import DataLoaderIter, DataLoader
from torch.autograd import Variable
import torch


class Solver(object):
    """docstring for Solver"""

    def __init__(self, model, data, **kwargs):
        super(Solver, self).__init__()
        print("init")
        self.model = model

        self.optimizer = kwargs.pop('optimizer', 'Adam')
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.loss = kwargs.pop('loss', '')

        if(len(kwargs) > 0):
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError(" Unrecognized arguments %s" % extra)

        if not hasattr(optim, self.optimizer):
            raise ValueError('Invalid optimizer "%s"' % self.optimizer)
        self.optimizer = getattr(optim, self.optimizer)(
            model.parameters(), **self.optim_config)

        self.loss_fn = self.loss()

        # Convert data to iter
        data_loader = DataLoader(data, batch_size=30, shuffle=True)
        self.data = DataLoaderIter(data_loader)
        self._reset()

    def _reset(self):
        """
        Setup some book-keeping variable for optimization.
        Don't call this manually.
        """

        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _step(self):

        input, target = self.data.next()
        input, target = Variable(input), Variable(
            target.type(torch.LongTensor).squeeze())
        output = self.model(input)
        loss = self.loss_fn(output, target)
        self.loss_history.append(loss)
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def train(self):

        num_train = len(self.data)
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in xrange(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print '(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations,
                    self.loss_history[-1].data.numpy()[0])

            epoch_end = (t + 1) % iterations_per_epoch == 0

            if epoch_end:
                self.epoch += 1
