import unittest

from pitchnet.io import wav_load, wav_save
import os
import shutil
import numpy as np


class TestWavIO(unittest.TestCase):
    def test_write_then_read(self):

        # Build audio and normalize
        expected_audio = np.array([x * 0.1 for x in range(100)])
        expected_audio /= expected_audio.max()

        temporary_filename = "__tmp_unittest_pitchnet_io.wav"

        # Write file
        wav_save(filename=temporary_filename, sample_rate=44100, samples=expected_audio)
        result_audio, result_samplerate = wav_load(temporary_filename, sample_rate=44100)

        delta = (result_audio - expected_audio) < 1e-6
        self.assertEqual(delta.all(), True)

        os.unlink(temporary_filename)


class TestDatasetIO(unittest.TestCase):
    def test_export_and_load_of_a_dataset(self):
        import pitchnet.dataset
        import pitchnet.io
        # Small dataset
        dt1 = pitchnet.dataset.MonophonicDataset(size=2, notes_per_sample=3, duration=0.5)

        # Export
        temporary_directory = "__tmp_unittest_pitchnet_io_datasets"
        pitchnet.io.export_dataset(dt1, folder=temporary_directory)

        # Load
        dt2 = pitchnet.io.import_dataset(folder=temporary_directory)

        # Assert equal samples
        a, b = dt1.__getitem__(0)
        x, y = dt2.__getitem__(0)
        delta_i, delta_j = a - x < 1e-12, y - b < 1e-12
        self.assertTrue(delta_i.all())
        self.assertTrue(delta_j.all())

        shutil.rmtree(temporary_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
