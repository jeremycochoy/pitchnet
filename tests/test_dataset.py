import unittest

import numpy as np
from pitchnet.dataset import MonophonicDataset, WavFolderDataset, WavSamplerDataset


class TestMonophonicDataset(unittest.TestCase):

    def test_two_generated_sample_are_different(self):
        dt = MonophonicDataset(seed=42, size=2, notes_per_sample=2)

        # Assert the audio signals are different
        a, b = dt.__getwav__(0)[0], dt.__getwav__(1)[0]
        if len(a) == len(b):
            self.assertNotAlmostEqual(dt.__getwav__(0)[0], dt.__getwav__(1)[0])
        else:
            self.assertNotEqual(len(a), len(b))

        # Get items:
        item_a, item_b = dt.__getitem__(0), dt.__getitem__(1)

        # Assert the frames are different
        a, b = item_a[0], item_b[0]
        if a.shape == b.shape:
            self.assertFalse((a - b < 1e-6).all())
        else:
            self.assertNotEqual(a.shape, b.shape)

        # Assert the descriptors are different
        a, b = item_a[1], item_b[1]
        if a.shape == b.shape:
            self.assertFalse(((a - b < 1e-6).all()))
        else:
            self.assertNotEqual(a.shape, b.shape)

    def test_generated_sample_are_reproducible(self):
        dt = MonophonicDataset(seed=42, size=20, notes_per_sample=2)
        index = np.random.randint(0, 20)

        # Assert reproducibility of the samples given one generator
        item_a, item_b = dt.__getitem__(index), dt.__getitem__(index)
        a, b = self._tuple_tensors_to_tuple_numpys(item_a, item_b)
        self.assertEqual(a[0].shape, b[0].shape)
        self.assertEqual(a[1].shape, b[1].shape)
        delta_frame = a[0]-b[0] < 1e-6
        delta_descriptor = a[1]-b[1] < 1e-6
        self.assertTrue(delta_frame.all())
        self.assertTrue(delta_descriptor.all())

        # Assert reproducibility of the samples given two generators
        dt2 = MonophonicDataset(seed=42, size=20, notes_per_sample=2)  # same size and seed
        item_b = dt2.__getitem__(index)
        a, b = self._tuple_tensors_to_tuple_numpys(item_a, item_b)
        self.assertEqual(a[0].shape, b[0].shape)
        self.assertEqual(a[1].shape, b[1].shape)
        delta_frame = a[0]-b[0] < 1e-6
        delta_descriptor = a[1]-b[1] < 1e-6
        self.assertTrue(delta_frame.all())
        self.assertTrue(delta_descriptor.all())

    def test_shape_of_dataset_are_right(self):
        dt = MonophonicDataset(seed=23, size=10, notes_per_sample=2)
        index = np.random.randint(0, 10)

        # Assert audio is mono
        audio, sample_rate = dt.__getwav__(index)
        self.assertEqual(len(audio.shape), 1)

        # Assert we have 3 dimensions : TxCxS
        frames, descriptors = dt.__getitem__(index)
        self.assertEqual(len(frames.shape), 3)

        # Assert S is 4096/2+1
        self.assertEqual(frames.shape[2], 4096 // 2 + 1)

        # Assert descriptor is a pair (onehot and segmentation)
        self.assertEqual(len(descriptors.shape), 2)  # Tx129
        self.assertEqual(descriptors.shape[1], 128 + 1)  # onehot(128) + segmentation(1)

        # Assert frames are Tx4xS
        self.assertEqual(frames.shape[1], 4)

    @staticmethod
    def _tuple_tensors_to_tuple_numpys(a, b):
        return tuple([x.numpy() for x in a]), tuple([x.numpy() for x in b])


class TestWavFolderDataset(unittest.TestCase):
    def test_load_a_file(self):
        from pitchnet.io import wav_save
        import math
        import shutil
        import os

        directory = "__tests_dataset_py_test_load_a_file"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        filename = directory + "/" + "__test_file.wav"
        signal = np.array([math.sin(x / 100) for x in range(1500)])
        wav_save(filename=filename, samples=signal, sample_rate=48000)

        dt = WavFolderDataset("__tests_dataset_py_test_load_a_file")

        # Check we have one item
        self.assertEqual(dt.size, 1)

        # Check we can parse it without crash
        dt._compute_item(0)

        # Do some cleanup
        shutil.rmtree(directory)


class TestWavSamplerDataset(unittest.TestCase):
    def test_one_can_build_and_run_an_instance(self):
        dt = WavSamplerDataset(sample_folder="./tests/data/voice_samples")
        signal, frames, descriptors = dt._compute_item(0)


if __name__ == '__main__':
    unittest.main()
