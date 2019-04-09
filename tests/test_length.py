import unittest
from dataloader import (
    StepSequentialSampler,
    StepRandomSampler,
    StepDataLoader,
)


class LengthTest(unittest.TestCase):
    def test_sampler(self):
        seq = list(range(10))

        sampler = StepSequentialSampler(seq)
        self.assertEqual(len(sampler), 10)

        sampler = StepSequentialSampler(seq, 3)
        self.assertEqual(len(sampler), 3)

        sampler = StepSequentialSampler(seq, 20)
        self.assertEqual(len(sampler), 20)

        sampler = StepRandomSampler(seq)
        self.assertEqual(len(sampler), 10)

        sampler = StepRandomSampler(seq, 3)
        self.assertEqual(len(sampler), 3)

        sampler = StepRandomSampler(seq, 20)
        self.assertEqual(len(sampler), 20)

    def test_dataloader(self):
        seq = list(range(10))

        dataloader = StepDataLoader(seq)
        self.assertEqual(len(dataloader), 10)

        dataloader = StepDataLoader(seq, num_samples=5)
        self.assertEqual(len(dataloader), 5)

        dataloader = StepDataLoader(seq, num_samples=20)
        self.assertEqual(len(dataloader), 20)

        dataloader = StepDataLoader(seq, shuffle=True)
        self.assertEqual(len(dataloader), 10)

        dataloader = StepDataLoader(seq, shuffle=True, num_samples=5)
        self.assertEqual(len(dataloader), 5)

        dataloader = StepDataLoader(seq, shuffle=True, num_samples=20)
        self.assertEqual(len(dataloader), 20)
