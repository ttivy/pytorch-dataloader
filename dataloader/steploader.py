import logging
from torch.utils.data import DataLoader
from .stepsampler import (
    StepRandomSampler,
    StepSequentialSampler
)

logger = logging.getLogger(__name__)


class StepDataLoader(DataLoader):
    r"""Extended DataLoader by steps
    """

    def __init__(self, dataset, num_samples=None, shuffle=False, **kwargs):
        kwargs.pop('sampler', None)

        self.num_samples = num_samples

        if shuffle:
            sampler = StepRandomSampler(dataset, num_samples)
        else:
            sampler = StepSequentialSampler(dataset, num_samples)

        super().__init__(dataset, sampler=sampler, **kwargs)
