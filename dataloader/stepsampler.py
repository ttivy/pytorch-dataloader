import logging
from abc import ABCMeta, abstractmethod
from itertools import islice
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class _StepSampler(Sampler, metaclass=ABCMeta):
    r"""Base class for Samplers by steps.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
    """

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples
        self._iter = None

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    @abstractmethod
    def _iter_inf(self):
        pass

    def __iter__(self):
        if self._iter is None:
            self._iter = self._iter_inf()
        return islice(self._iter, self.num_samples)

    def __len__(self):
        return self.num_samples


class StepSequentialSampler(_StepSampler):
    r"""Samples elements sequentialy by steps.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
    """

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source, num_samples)

    def _iter_inf(self):
        logger.debug('Started infinite iteration')
        n = len(self.data_source)
        while True:
            yield from range(n)


class StepRandomSampler(_StepSampler):
    r"""Samples elements randomly by steps.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
    """

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source, num_samples)

    def _iter_inf(self):
        logger.debug('Started infinite iteration')
        n = len(self.data_source)
        while True:
            yield from torch.randperm(n).tolist()
