import unittest
from dataloader import (
    StepSequentialSampler,
    StepRandomSampler,
    StepDataLoader,
)


class IterTest(unittest.TestCase):
    def test_sampler(self):
        seq = list(range(3))

        sampler = StepSequentialSampler(seq)
        self.assertCountEqual(sampler, [0, 1, 2])

        sampler = StepSequentialSampler(seq, 2)
        self.assertCountEqual(sampler, [0, 1])

        sampler = StepSequentialSampler(seq, 5)
        self.assertCountEqual(sampler, [0, 1, 2, 0, 1])

        sampler = StepRandomSampler(seq)
        self.assertCountEqual(sorted(sampler), [0, 1, 2])

    def test_dataloader(self):
        seq = list(range(3))

        dataloader = StepDataLoader(seq)
        self.assertCountEqual(map(int, dataloader), [0, 1, 2])

        dataloader = StepDataLoader(seq, num_samples=2)
        self.assertCountEqual(map(int, dataloader), [0, 1])

        dataloader = StepDataLoader(seq, num_samples=5)
        self.assertCountEqual(map(int, dataloader), [0, 1, 2, 0, 1])

        dataloader = StepDataLoader(seq, shuffle=True)
        self.assertCountEqual(map(int, dataloader), [0, 1, 2])
