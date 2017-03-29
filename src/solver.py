from torch import optim


class Solver(object):
    """docstring for Solver"""

    def __init__(self, model, data, **kwargs):
        super(Solver, self).__init__()
        self.model = model
        self.data = data

        self.optimizer = kwargs.pop('optimizer', 'Adam')
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', '')

        if(len(kwargs) > 0):
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError(" Unrecognized arguments %s" % extra)

        if not hasattr(optim, self.optimizer):
            raise ValueError('Invalid optimizer "%s"' % self.optimizer)
        self.optimizer = getattr(optim, self.optimizer)(
            model.parameters(), **self.optim_config)

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

        self.optim_config = {}

        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_config[p] = d

    def _step(self):

        input, output = self.data.get()
        loss = self.model.loss(input, output)
        self.loss_history.append(loss)
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def train(self):

        num_train = self.data.size()
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in xrange(num_iterations):
            self._step()

            if self.verbose and t % self.print_every:
                print '(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1])

            epoch_end = (t + 1) % iterations_per_epoch == 0

            if epoch_end:
                self.epoch += 1
