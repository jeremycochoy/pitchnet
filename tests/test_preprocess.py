import unittest

from pitchnet.preprocess import pitch_to_onehot, onehot_to_pitch, signal_to_sdft, \
    signal_to_frames, downsample_signal_to_frame_scale, onehot_to_pitch_torch2
import numpy as np


class TestOneHotMethods(unittest.TestCase):
    def test_bounds_checking(self):
        with self.assertRaises(Exception):
            pitch_to_onehot(2, vector_length=2)

    def test_intergers_values(self):
        self.assertTrue(np.all(pitch_to_onehot(0, vector_length=2) == np.array([1, 0])))
        self.assertTrue(np.all(pitch_to_onehot(1, vector_length=2) == np.array([0, 1])))

    def test_half_values(self):
        # Assert one hot at position 1/2 : 0.5 and 0.5
        self.assertTrue(np.all(pitch_to_onehot(0.5, vector_length=2) == np.array([0.5, 0.5])))
        self.assertTrue(np.all(pitch_to_onehot(1.5, vector_length=4) == np.array([0, 0.5, 0.5, 0])))

    def test_composition_invariant(self):
        for i in range(10):
            r = np.random.randint(0, 128)
            self.assertEqual(onehot_to_pitch(pitch_to_onehot(r)), r)

    def test_implementation_equivalence(self):
        for i in range(10):
            r = np.random.rand(128)
            import torch
            a = onehot_to_pitch_torch2(torch.Tensor(r)[None, None, :]).numpy()[0, 0]
            b = onehot_to_pitch(r)
            self.assertAlmostEquals(a, b, delta=1e-5)

        r = np.random.rand(2, 1, 128)
        import torch
        a = onehot_to_pitch_torch2(torch.Tensor(r)).numpy()
        b0 = onehot_to_pitch(r[0, 0])
        b1 = onehot_to_pitch(r[1, 0])
        self.assertAlmostEquals(a[0, 0], b0, delta=2e-5)
        self.assertAlmostEquals(a[1, 0], b1, delta=2e-5)


class TestSDFTMethods(unittest.TestCase):
    def test_sdft_shape(self):
        signal = np.array(range(300), dtype=float)
        sdft = signal_to_sdft(signal, frame_length=1024, hop_length=1)
        # Assert shape T x S
        self.assertEqual(tuple(sdft.shape), tuple([300 + 1, 1024 / 2 + 1]))

    def test_sdft_amplitude(self):
        from math import sin, pi
        # We produce a signal with strong amplitude in bin 42
        signal = np.array([sin(x * 100 * 2 * pi / 1024) for x in range(100)])
        sdft = signal_to_sdft(signal, frame_length=1024, hop_length=1)

        result = np.argmax(np.abs(sdft), 1)
        expected = np.repeat(100, 101)
        #  Assert strongest bin is 100 for each frame
        self.assertEqual(len(result), len(expected))
        for i in range(len(result)):
            self.assertEqual(result[i], expected[i])

    def test_frames_amplitude(self):
        from math import sin, pi
        # We produce a signal with strong amplitude in bin 42
        signal = np.array([sin(x * 100 * 2 * pi / 1024) for x in range(100)])
        frames = signal_to_frames(signal, frame_length=1024, hop_length=1)

        result_amplitude = np.argmax(frames[:, 0, :], 1)
        expected_amplitude = np.repeat(100, 101)

        #  Assert strongest bin is 100 for each frame
        self.assertEqual(len(result_amplitude), len(expected_amplitude))
        for i in range(len(result_amplitude)):
            self.assertEqual(result_amplitude[i], expected_amplitude[i])

    def test_sdft_output(self):
        # Test the output on a previously computed result,
        # to detect any change in the algorithm.

        signal = np.array([1, 2, 3, 4, 5], dtype=float)
        sdft = signal_to_sdft(signal, frame_length=8, hop_length=1)

        expected_result = [
            [2.7377706+0.j, -1.8149372+1.4292182j, 0.38571915-1.222521j, -0.08603159+0.29968753j, 0.2927288+0.j],
            [5.4382553+0.j, -3.8697422+1.3776057j, 1.1479485-0.88329697j, 0.06780438-0.12843512j, -0.13027658+0.j],
            [8.75+0.j, -5.924547+0.7147327j, 1.2989173-0.544073j, 0.22164036+0.05470268j, 0.05797853+0.j],
            [11.120469+0.j, -7.8462353-1.2107873j, 2.5794168-0.39310414j, 0.24235988+1.2342546j, -1.071552+0.j],
            [10.764651+0.j, -7.1745653-4.78839j, 2.9186409+3.4254274j, -2.330279-1.1208274j, 2.4077556+0.j],
            [7.7622294+0.j, -2.9611206-6.204861j, -2.445042+4.187657j, 2.9611206-1.3147775j, -2.8721457+0.j]
        ]

        # Check the sdft is at less than 1e-6 from the expected result
        self.assertTrue(np.all(np.abs(expected_result - sdft) < 1e-6))

    def test_downsample_signal_to_frame_scale(self):
        for _ in range(100):
            # Create a dummy signal, framesize, hop
            frame_length = np.random.randint(1, 1024+1)
            hop_length = np.random.randint(1, 65)
            signal = np.zeros(np.random.randint(10, 1500))

            # Process through sdft and downsample
            frames = signal_to_sdft(signal, frame_length=frame_length, hop_length=hop_length)
            downsampled = downsample_signal_to_frame_scale(signal, frame_length=frame_length, hop_length=hop_length)

            self.assertEqual(len(frames), len(downsampled))


if __name__ == '__main__':
    unittest.main()
