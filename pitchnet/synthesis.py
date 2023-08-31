import numpy as np
import scipy.signal

__all__ = ['Note', 'Synthesizer', 'segmentation_to_onset', 'audio_to_descriptors', 'wav_file_to_descriptors']


class Note:
    """
    A note is a midi note defined by
      - a midi number `note_number`
      - a `position` in seconds
      - a `duration` in seconds
    """

    def __init__(self):
        self.number = 69
        self.position = 0
        self.duration = 0
        self.confidence = 1

    def __repr__(self):
        out = ["<Note number:", str(self.number), " ",
               "position:", str(self.position), " ",
               "duration:", str(self.duration), " ",
               "confidence:", str(self.confidence), ">"]
        return ''.join(out)



class Synthesizer:
    """
    A generator for synthesizing audio and midi data.

    You can control the seed of the whole generator by calling `set_seed`.
    """

    seed = None
    numpy_generator = None
    voice_samples = None

    def __init__(self, seed=None, voice_samples=None):
        if seed is None:
            self.set_seed()
        else:
            self.set_seed(seed)

        self.set_voice_samples_path(voice_samples)

    def set_seed(self, seed=42):
        self.seed = seed
        self.numpy_generator = np.random.RandomState(self.seed)

    def generate_note_sequence(self, number_of_notes=20, min_duration=40, max_duration=1500,
                               silence_min_duration=10, silence_max_duration=300,
                               silence_probability=0.3, total_duration=0, integer_notes=False, min_pitch=21, max_pitch=108):
        """
        Generate a random sequence of notes
        (called a melody). Notes will be spaced by silences.
        Durations are inclusive.

        :param min_duration: Min duration of a note in ms
        :param max_duration: Max duration of a note in ms
        :param silence_min_duration: Min duration of the silence in ms
        :param silence_max_duration: Max duration of the silence in ms
        :param number_of_notes: Number of notes in the melody.
        :param total_duration: Maximal duration of the full sequence, expressed in seconds. 0 means no limit.
        :return: A list of Note
        """
        output = []
        scale = 0.001  # 10ms Scale of information used to store notes
        time = 0
        for i in range(number_of_notes):
            note = Note()
            if integer_notes:
                note.number = self.numpy_generator.randint(min_pitch, max_pitch)
                if len(output) > 1:
                    while note.numer == output[-1].number:
                        note.number = self.numpy_generator.randint(min_pitch, max_pitch)
            else:
                note.number = self.numpy_generator.uniform(min_pitch, max_pitch)
            note.position = time
            note.duration = self.numpy_generator.randint(min_duration, max_duration + 1) * scale
            output += [note]
            time += note.duration

            # Break if the note allow us to reach the maximal sequence duration
            if total_duration and time > total_duration:

                delta = time - total_duration
                note.duration -= delta

                # If the note is too small, remove it
                if note.duration < min_duration:
                    output = output[:-1]

                # Leave the loop
                break

            # Add silence with a given probability
            if self.numpy_generator.uniform() < silence_probability:
                time += self.numpy_generator.randint(silence_min_duration, silence_max_duration + 1) * scale

            # Break if the silence makes us reach the maximal sequence duration
            if total_duration:
                if time > total_duration:
                    break
        return output

    @staticmethod
    def low_pass_filter(signal, freq=4410, sample_rate=44100):
        """
        This is a filter of order N and of critical frequency Wn
        It takes effect around 4410Hz (sample_rate * 0.1)

        More documentation on IIR numeric filter
        at http://www.f-legrand.fr/scidoc/docmml/numerique/filtre/filtrenum/filtrenum.html

        :param signal: Input audio signal
        :param freq: The cut off frequency of the filter
        :param sample_rate: Sample rate of the audio signal
        :return: The filtered signal
        """

        b, a = scipy.signal.butter(N=2, Wn=[freq/sample_rate*2], btype='lowpass')
        filtered_signal = scipy.signal.lfilter(b, a, signal)
        return filtered_signal

    def notes_to_monophonic_signal(self, notes, sample_rate=44100, frequency_noise_amplitude=0.0,  synthesizer=None):
        """
        Take a list of notes and convert them to audio sampled at `sample_rate`.

        Default sampling rate of synthesizer is 44100

        :param notes: A list of `Note` objects that will be used to generate the audio.
        :param synthesizer: Control the shape of the wave.
                            It can be 'sine', 'triangle', 'square', 'sawtooth', or
                            simply a synthesizer object. It can be set to None for default value (sawtooth).
        :param sample_rate: Simpling frequency (number of samples per seconds) used for the signal. Default to 44100 Hz.
        :param frequency_noise_amplitude: Amplitude of additive noise in semitone, constant along a single note.
        :return: A audio signal as a np.array sampled at rate sample_rate.
        """
        from synthesizer import Synthesizer, Waveform

        synthesizer_arr = ['sine', 'triangle', 'square', 'sawtooth']
        if synthesizer == 'random':
            synthesizer = self.numpy_generator.choice(synthesizer_arr)

        if synthesizer == 'sine':
            synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)
        if synthesizer == 'triangle':
            synthesizer = Synthesizer(osc1_waveform=Waveform.triangle, osc1_volume=1.0, use_osc2=False)
        if synthesizer == 'square':
            synthesizer = Synthesizer(osc1_waveform=Waveform.square, osc1_volume=1.0, use_osc2=False)
        elif synthesizer == 'sawtooth' or synthesizer is None:
            synthesizer = Synthesizer(osc1_waveform=Waveform.sawtooth, osc1_volume=1.0, use_osc2=False)
            if sample_rate != 44100:
                raise Exception("The default synthesizer can only handle sampling at 44100Hz.")

        output = []
        time = 0
        for note in notes:
            # Insert silence
            output += [np.zeros(int(sample_rate * (note.position - time)))]
            # Compute note frequency (with noise)
            number = note.number + self.numpy_generator.normal(0, frequency_noise_amplitude, 1)[0]
            frequency = 440 * pow(2, (number - 69) / 12)
            # Compute note wave
            assert frequency > 1  # Sanity check: audible frequencies are > 1Hz.
            duration = note.duration  # duration in seconds
            signal = synthesizer.generate_constant_wave(frequency, duration)
            output += [signal]
            # Remember time elapsed to the current output.
            time = note.position + note.duration
        output = np.concatenate(output)

        return output

    @staticmethod
    def notes_to_descriptor(notes, sample_rate=44100):
        """
        Build curve descriptor from notes.

        First descriptor is the pitch expressed in midi numbers and
         function of the time.

        The second descriptor is a collection of bounding boxes defined by
        1) the width expressed in log(1 + milliseconds)
        2) the offset expressed in log(1 + milliseconds)

        :param notes: A list of `Note` object to generate the descriptors.
        :param sample_rate: The number of samples per seconds.
        :return: A pair of descriptors np.array([output_pitch, output_segment_width, output_segment_offset])
        """
        time = 0
        output_pitch = []
        output_segment_width = []
        output_segment_offset = []
        for note in notes:
            # Insert silence
            duration_in_samples = int(sample_rate * (note.position - time))
            output_pitch += [np.zeros(duration_in_samples)]
            output_segment_width += [np.zeros(duration_in_samples)]
            output_segment_offset += [np.zeros(duration_in_samples)]
            # Inset note
            duration_in_samples = int(sample_rate * note.duration)
            output_pitch += [np.array([note.number]).repeat(duration_in_samples)]
            output_segment_width += [np.ones(duration_in_samples) * note.duration]
            output_segment_offset += [(np.cumsum(np.ones(duration_in_samples)) - 1) / (duration_in_samples - 1)]
            # Update time
            time = note.position + note.duration

        output_pitch = np.concatenate(output_pitch)
        output_segment_width = np.concatenate(output_segment_width)
        output_segment_offset = np.concatenate(output_segment_offset)

        return np.array([output_segment_width, output_segment_offset, output_pitch])

    def generate_random_pair(self, number_of_notes=500, sample_rate=44100, frequency_noise_amplitude=0.1,
                             static_noise_amplitude=0.1, synthesizer=None, lowpass_filter_probability=0,
                             max_duration=0):
        """
        Generate a random pair of (input, output) for training a network
        on a wav input / segmented output.

        This is monophonic audio (one note at a time).

        The output is a (pitch: np.array, is_note_on: np.array).
        The first vector is the pitch estimated function of time,
        the second is the signed probability a note is being played (0 or ±1 for training).
        The sign alternate between two consecutive notes, so that note segmentation
        is still obvious when there is no silence between two note changes.

        :param lowpass_filter_probability: Probability to activate low pass.
        :param static_noise_amplitude: Amplitude of static noise.
        :param frequency_noise_amplitude: Amplitude of perturbation of pitch.
        :param synthesizer: Synthesizer used to produce sound, See notes_to_monophonic_signal doc.
        :param sample_rate: Sample rate of the produced signal.
        :param number_of_notes: Number of notes to generate in this sample
        :param max_duration: Maximum duration of a note in seconds
        :return: A pair (audio signal, descriptor).
        """

        # Input notes
        notes = self.generate_note_sequence(number_of_notes=number_of_notes, total_duration=max_duration)

        # The input signal for training
        signal = self.notes_to_monophonic_signal(
            notes, synthesizer=synthesizer, sample_rate=sample_rate,
            frequency_noise_amplitude=frequency_noise_amplitude
        )
        # Add noise and fx
        signal = self.add_fx(
            signal, lowpass_filter_probability=lowpass_filter_probability,
            static_noise_amplitude=static_noise_amplitude,
            sample_rate=sample_rate
        )

        # Generate output expected by the network
        descriptors = self.notes_to_descriptor(notes, sample_rate=sample_rate)

        return signal, descriptors

    def add_fx(self, signal, lowpass_filter_probability=0.3, static_noise_amplitude=0.1,
               low_pass_freq=(1000, 20000), sample_rate=44100):
        """

        Apply fx after normalizing the signal.

        :param sample_rate: Sample rate (number of samples per seconds) of input signal.
        :param signal: Input audio signal to processed.
        :param lowpass_filter_probability: Probability of activating the low pass filter.
        :param static_noise_amplitude: Amplitude of additive white noise added to the output signal.
        :param low_pass_freq: Range interval of frequency for low pass. Chosen from using an uniform distribution.
        :return: Signal with effects.
        """

        # Normalize signal just in case:
        signal = np.array(signal) / np.abs(signal).max()

        if lowpass_filter_probability and self.numpy_generator.uniform() < lowpass_filter_probability:
            freq = self.numpy_generator.uniform(*low_pass_freq)
            signal = self.low_pass_filter(signal, freq=freq, sample_rate=sample_rate)

        # Add static noise
        signal += self.numpy_generator.normal(0, static_noise_amplitude, len(signal))

        return signal

    def set_voice_samples_path(self, path=None):
        """
        Change the directory from which the
        voice samples used by notes_to_monophonic_stretched_voice are loaded.

        :param path: Path of the folder containing the samples to be used.
        """
        if path is None:
            path = './data/voice_samples'
        self.voice_samples = path

    def notes_to_monophonic_stretched_voice(self, notes, sample_rate=44100, threshold=0.25):
        """
        To change the sample directory, call `set_voice_samples_path` with your path.
        By default, the directory is './data/voice_samples'.

        This script ignore the pitch (midi number) of the note.

        :param notes: A list of `Notes` that will be used to generate the audio (only timing)
        :param sample_rate: Simpling frequency used for the signal. Default to 44100 Hz.
        :param threshold: Threshold to consider the audio from a sample to be non silence. Used to
                            remove silence at extremities of files.
        :return: A signal: np.array sampled at sample_rate.
        """
        from .stretch import stretch
        from .io import wav_load
        from glob import glob
        import scipy.signal

        # Samples used for making notes
        samples = []
        for ext in ('.wav', '.WAV'):
            samples.extend(glob(self.voice_samples + "/**/*" + ext, recursive=True))
        if not samples:
            raise Exception(f"Cannot find samples in {self.voice_samples}")

        output = []
        time = 0
        for note in notes:
            # Insert silence
            output += [np.zeros(int(sample_rate * (note.position - time)))]

            # Get a sample and load it
            sample = self.numpy_generator.choice(samples)
            sample, _ = wav_load(sample)
            sample /= np.abs(sample).max()

            # Skip 'silence' on both side
            noisy = np.where(np.abs(sample) >= threshold)[0]
            sample = sample[noisy[0]:noisy[-1]]

            # Cut the sample in 3 pieces
            cut_a, cut_b = len(sample) * 1 // 3, len(sample) * 2 // 3
            start = sample[:cut_a]
            middle = sample[cut_a:cut_b]
            end = sample[cut_b:]

            # Compute timing in samples
            duration = round(note.duration * sample_rate)
            # Audio is smaller or same size than the note, stretch the whole file to fit
            if len(sample) >= duration - sample_rate * 0.010:
                signal = stretch(sample, len(sample) / duration)
            # Stretch the middle sample to match the note duration
            else:
                middle_duration = duration - len(start) - len(end)
                middle = stretch(middle, len(middle) / middle_duration)
                signal = np.concatenate([start, middle, end])

            # Resize the sample (stretch do not control pricisely the number of samples
            signal = scipy.signal.resample_poly(signal, duration, len(signal))
            assert len(signal) == duration
            output += [signal]

            # Remember time elapsed to the current output.
            time = note.position + note.duration
        output = np.concatenate(output)

        # Normalize
        output /= np.abs(output).max()
        return output


def segmentation_to_onset(note_segmentation, decay=0):
    """
    Convert segmentation curve to the onset curve (with an optional exponential decay).
    Hint: A 0.2 exponential decay looks good.

    :param note_segmentation: Input np.array containing 1, 0, -1 for note segmentation
    :param decay: Exponential decay applied to produced signal. By default the signal is "perfect".
    :return: A curve with spike whenever a new note is played, a second whenever a note finish.
    """

    # Add a one sample silence at the begining
    note_segmentation = np.concatenate([[0], note_segmentation])

    # Separate positive and negative part of the curve and take abs value.
    segment_a = note_segmentation.clip(0, 1)
    segment_b = -note_segmentation.clip(-1, 0)

    # Compute the derivative of positive / negative curve
    delta_a = segment_a[1:] - segment_a[:-1]
    delta_b = segment_b[1:] - segment_b[:-1]

    # Combines derivatives : Positive spike is a note start, negative spike is a note end.
    starts = delta_a.clip(min=0) + delta_b.clip(min=0)
    ends = delta_a.clip(max=0) + delta_b.clip(max=0)

    # Apply decay if required
    if decay > 0:
        def decay_filter(s):
            out = np.zeros(s.shape)
            out[0] = s[0]
            for i in range(1, len(s)):
                out[i] = s[i] + decay * out[i-1]
            return out
        starts, ends = decay_filter(starts), decay_filter(ends)

    return starts, ends


def audio_to_descriptors(audio, input_sample_rate=48000, output_sample_rate=44100):
    """
    Parse an audio and analyse it using the v2p algorithm.

    :param audio: The input audio signal.
    :param input_sample_rate: Sample rate of the audio input. Default is 48kHz.
    :param output_sample_rate:  Sample rate of the audio output. Default is 44.1kHz.
    :return: A pair (audio, descriptors) both expressed from the same sample_rate.
    """
    import v2p
    import scipy.signal as sps

    # Check the input sample rate
    if input_sample_rate != 48000:
        raise Exception("The v2p python binding support only a sample rate of 48000Hz.")

    # Pitch expressed in midi numbers
    pitch = v2p.boersma_path(audio)
    midi_numbers = v2p.pitch_to_midi_numbers(pitch)

    # Segmentation
    notes = v2p.midi_numbers_to_notes(midi_numbers)
    segmentation = np.array(0).repeat(len(pitch))  # silence (note = 0)
    i = 0
    sign = 1
    for note in notes:
        if note.note_number < 1:
            continue
        segmentation[int(note.position):int(note.position + note.duration)] = np.array(sign).repeat(1)
        sign *= -1
        i += 1

    # Re-sample curves to `output_sample_rate`
    def rsz(signal):
        signal = [np.repeat(p, output_sample_rate / 100) for p in signal]
        signal = np.concatenate(signal)
        return signal
    midi_numbers = rsz(midi_numbers)
    segmentation = rsz(segmentation)

    # Re-sample audio signal to 44.1kHz
    audio = sps.resample_poly(audio, up=len(midi_numbers), down=len(audio))

    assert len(audio) == len(midi_numbers)
    assert len(midi_numbers) == len(segmentation)

    descriptor = np.array([midi_numbers, segmentation])
    return audio, descriptor


def wav_file_to_descriptors(filename, duration):
    """
    Open a load a wav file and process it with the v2p algorithm.
    See `audio_to_descriptors` for more details.

    The result is a pair (audio, descriptors).

    :param filename: Path the the wav file loaded.
    :param duration: Maximal duration of the audio expressed in seconds.
    :return: A pair (audio, descriptors).
    """
    from pitchnet.io import wav_load
    audio, sample_rate = wav_load(filename, sample_rate=48000, duration=duration)
    return audio_to_descriptors(signal=audio, input_sample_rate=48000, output_sample_rate=44100)
