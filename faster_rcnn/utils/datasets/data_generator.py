import threading
import numpy as np
import multiprocessing
import time
import Queue
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Iterator(object):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size=1, shuffle=True, seed=10):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            logger.debug("_flow_index")
            yield (index_array[current_index: current_index + current_batch_size], current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class CocoGenerator(Iterator):
    def __init__(self, data,
                 batch_size=1, shuffle=False, seed=None, sorted_index=None):
        self.data = data
        super(CocoGenerator, self).__init__(
            len(data), batch_size, shuffle, seed)

        self.sorted_index = sorted_index

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                if self.sorted_index is None:
                    index_array = np.arange(n)
                else:
                    index_array = self.sorted_index

                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size], current_index, current_batch_size)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        logger.debug('Next executed')
        logger.debug('hehehe' + str(index_array))
        logger.debug(current_index)
        logger.debug(current_batch_size)

        data = [self.data[i] for i in index_array]
        data = [i for i in data if i is not None]
        logger.debug([d['im_name'] for d in data])
        return data


class Enqueuer(object):
    def __init__(self, generator, use_multiprocessing=False, shuffle=False, wait_time=0.05, random_seed=None):
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self.queue = None
        self._stop_event = None
        self._threads = []
        self.wait_time = wait_time
        self.random_seed = random_seed

        if self._use_multiprocessing:
            self.lock = multiprocessing.Lock()
        else:
            self.lock = threading.Lock()

    def start(self, workers=3, max_queue_size=10):
        logger.debug('start')

        def data_generator_task():
            logger.debug("task start")
            logger.debug("_stop_event %s" % str(self._stop_event.is_set()))
            while not self._stop_event.is_set():
                try:
                    logger.debug("Queue size %d " % self.queue.qsize())
                    logger.debug("use_multiprocessing %s" %
                                 str(self._use_multiprocessing))
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            logger.debug("try")
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = Queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(
                        target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1

                else:
                    thread = threading.Thread(target=data_generator_task)

                self._threads.append(thread)
                thread.start()
        except Exception as e:
            logger.debug(e)
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)
        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        while self.is_running():
            logger.debug("Next")
            logger.debug("queue.empty : %s" % str(self.queue.empty()))
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
            pass
