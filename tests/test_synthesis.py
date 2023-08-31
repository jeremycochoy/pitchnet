import unittest
import numpy as np

from pitchnet.synthesis import Synthesizer, Note


class TestSynthesizerNotes(unittest.TestCase):
    gen = Synthesizer()

    def test_note_generation_silence(self):
        # Generate silence only
        silence_len = np.random.randint(10, 30)
        notes = self.gen.generate_note_sequence(number_of_notes=2, min_duration=1, max_duration=5,
                                                silence_min_duration=silence_len, silence_max_duration=silence_len,
                                                silence_probability=1)
        # Expect two notes
        self.assertEqual(len(notes), 2)
        # Expect a silence of `silence_len` between two notes
        delta_silence = notes[1].position - (notes[0].position + notes[0].duration)
        self.assertAlmostEqual(delta_silence, silence_len / 1000)

    def test_note_generation_sequence(self):
        # Generate notes
        note_duration = np.random.randint(10, 30)
        notes = self.gen.generate_note_sequence(number_of_notes=10, min_duration=note_duration,
                                                silence_min_duration=0, silence_max_duration=0,
                                                max_duration=note_duration)

        duration_sum = 0
        for note in notes:
            # Assert note have fixed length
            self.assertAlmostEqual(note.duration, note_duration / 1000)
            # Assert no silence
            self.assertAlmostEqual(note.position - duration_sum, 0)
            duration_sum += note.duration

    def test_note_generation_randomness(self):
        # Generate notes without silence
        a = self.gen.generate_note_sequence(number_of_notes=5)
        b = self.gen.generate_note_sequence(number_of_notes=5)
        # The two generated sequences should be different
        self.assertFalse(len(a) == len(b) and str(a) == str(b))


class TestSynthesizerNote(unittest.TestCase):
    gen = Synthesizer()

    def test_notes_to_monophonic(self):
        note = Note()
        note.number = 69  # A4
        note.position = 10  # 10ms
        note.duration = 10  # 30 ms
        sample_rate = 44100

        # Generate audio without noise
        audio = self.gen.notes_to_monophonic_signal([note], sample_rate=sample_rate,
                                                    frequency_noise_amplitude=0, synthesizer='square')

        # Assert duration is 10 + 10 ms
        self.assertEqual(len(audio), (note.position + note.duration) * sample_rate)

        # Assert no sound is present before 10ms
        audio_start_index = int(10 * sample_rate)
        audio_start = audio[:audio_start_index]
        self.assertAlmostEqual(np.abs(audio_start).sum(), 0)

        # Assert square wav (only +1 or -1 values)
        audio_end = audio[audio_start_index:]
        self.assertTrue((np.abs(audio_end) - 1 < 1e-6).all())


if __name__ == "__main__":
    unittest.main()
