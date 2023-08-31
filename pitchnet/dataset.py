import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from .synthesis import Synthesizer
from .preprocess import signal_to_frames, pitch_segmentation_to_segmentation_onehot, downsample_signal_to_frame_scale
from .io import wav_load

__all__ = ['MonophonicDataset',
           'WavFolderDataset',
           'WavSamplerDataset',
           'DatasetReader']


class MonophonicDataset(Dataset):
    """
    Generate parametrically a sequence of notes played by a synthesizer.

    The `__getitem__` methods return a pair `(x, y)` where
    `x` is a vector of shape Tx3xS (time x frames x freqs)
    and `y` is of shape Tx129 (1 for segmentation and 128 for midi numbers).

    The audio contain white noise.
    """

    def __init__(self, size=100, notes_per_sample=15, frequency_noise_amplitude=0.1, static_noise_amplitude=0.1,
                 seed=42, synthesizer='random', lowpass_filter_probability=0.3, duration=None, frame_length=4096):
        """

        :param size: Number of samples in the dataset
        :param notes_per_sample: Number of notes in each sample
        :param frequency_noise_amplitude: Additive noise to the frequency of notes
        :param static_noise_amplitude: Additive static noise to the sample
        :param seed: Random seed used to generate this dataset
        :param synthesizer:
        :param lowpass_filter_probability:
        """
        self.seed = seed
        self.size = size
        self.notes_per_sample = notes_per_sample
        self.frequency_noise_amplitude = frequency_noise_amplitude
        self.static_noise_amplitude = static_noise_amplitude
        self.synthesizer = synthesizer
        self.lowpass_filter_probability = lowpass_filter_probability
        self.duration=duration
        self.frame_length = frame_length

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = self._compute_item(index)[1:]

        return torch.Tensor(x), torch.Tensor(y)

    def _compute_item(self, index):
        # Setup random generators
        local_seed = index * len(self) + self.seed
        synthesizer = Synthesizer(seed=local_seed)

        signal, descriptors = synthesizer.generate_random_pair(
            number_of_notes=self.notes_per_sample,
            sample_rate=44100, frequency_noise_amplitude=self.frequency_noise_amplitude,
            lowpass_filter_probability=self.lowpass_filter_probability,
            static_noise_amplitude=self.static_noise_amplitude,
            max_duration=self.duration or 0
        )

        frames, descriptors = _signal_descriptors_to_frames_descriptors(
            signal=signal, descriptors=descriptors, frame_length=self.frame_length, hop_length=441
        )

        signal, frames, descriptors = _duration_uniformiser(self, signal, frames, descriptors)

        return signal, frames, descriptors

    def __getwav__(self, index):
        """
        Return a wav sample

        :param index:
        :return: A pair signal: np.array and sample_rate
        """
        # Return the wav (signal) sample
        signal = self._compute_item(index)[0]
        return signal, 44100


class WavFolderDataset(Dataset):
    def __init__(self, folder, duration=None, size=None, frame_length=4096):
        """
        Load a directory as a collection of audio input. This provide a MonophonicDataset without segmentation.
        The pitch detection rely on v2p algorith.

        :param folder: Folder where wav files are located.
        :param duration: Maximal time at which the audio is cut. Expressed in seconds. Default to None.
        """

        # Used to limit the temporal length of samples
        self.duration = duration
        self.sample_rate = 44100
        self.frame_length = frame_length

        if not os.path.isdir(folder):
            print("Could not open folder", folder)
            self.files = []
            self.size = 0
            return

        # Recursive file finding
        dir_list = glob.glob(folder + "/**", recursive=True)
        self.files = []
        for file in dir_list:
            if file.endswith(".wav") or file.endswith(".WAV"):
                self.files.append(file)
        self.files.sort()

        # Set dataset size
        if size is None:
            self.size = len(self.files)
        else:
            self.size = min(len(self.files), size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = self._compute_item(index)[1:]

        return torch.Tensor(x), torch.Tensor(y)

    def _compute_item(self, index):
        # Load audio
        audio, sample_rate = wav_load(self.files[index], 48000, self.duration)
        audio = audio * 1.0 # Fix a bug in scipy where int array are not handled correctly https://github.com/scipy/scipy/issues/15620

        # Convert raw audio to down-sampled audio and descriptors
        signal, descriptors = _audio_to_audio_and_descriptors(
            audio, sample_rate=sample_rate, duration=self.duration
        )

        frames, descriptors = _signal_descriptors_to_frames_descriptors(
            signal=signal, descriptors=descriptors, frame_length=self.frame_length, hop_length=441
        )

        signal, frames, descriptors = _duration_uniformiser(self, signal, frames, descriptors)

        return signal, frames, descriptors

    def __getwav__(self, index):
        """
        Return a wav sample

        :param index:
        :return: A pair signal: np.array and sample_rate
        """
        # Return the wav (signal) sample
        signal = self._compute_item(index)[0]
        return signal, 44100


class WavSamplerDataset(Dataset):
    """
    This dataset take wav files, basically words with a pitch, stretch them and glue them together
    into a pattern defined by a note sequence. Then, the audio is analysed through
    v2p to build the dataset labels.
    """
    def __init__(self, notes_per_sample=10, size=100, static_noise_amplitude=0.1,
                 sample_folder=None, lowpass_filter_probability=0.3, seed=42, duration=None, frame_length=4096):
        self.static_noise_amplitude = static_noise_amplitude
        self.notes_per_sample = notes_per_sample
        self.lowpass_filter_probability = lowpass_filter_probability
        self.size = size
        self.seed = seed
        self.sample_folder = sample_folder
        self.duration = duration
        self.frame_length = frame_length

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = self._compute_item(index)[1:]

        return torch.Tensor(x), torch.Tensor(y)

    def _compute_item(self, index):
        # Setup random generators
        local_seed = index * len(self) + self.seed
        synthesizer = Synthesizer(seed=local_seed, voice_samples=self.sample_folder)

        # Generate the audio and compute its segmentation
        audio, note_based_descriptors = self._generate_audio_and_descriptors(synthesizer)

        # Convert raw audio to down-sampled audio and v2p generated descriptors
        signal, audio_based_descriptors = _audio_to_audio_and_descriptors(
            audio, sample_rate=48000
        )

        # Replace the v2p segmentation by the generated semgentation
        descriptors = audio_based_descriptors
        print(audio_based_descriptors.shape, note_based_descriptors.shape)
        descriptors[0] = self._downscale_signal_to_given_length(note_based_descriptors[0], len(descriptors[0]))
        descriptors[1] = self._downscale_signal_to_given_length(note_based_descriptors[1], len(descriptors[0]))

        frames, descriptors = _signal_descriptors_to_frames_descriptors(
            signal=signal, descriptors=descriptors, frame_length=self.frame_length, hop_length=441
        )

        signal, frames, descriptors = _duration_uniformiser(self, signal, frames, descriptors)

        return signal, frames, descriptors

    def __getwav__(self, index):
        return self._compute_item(index)[0], 44100

    def _generate_audio_and_descriptors(self, synthesizer):
        # Generate audio
        notes = synthesizer.generate_note_sequence(number_of_notes=self.notes_per_sample, min_duration=100, total_duration=self.duration)
        audio = synthesizer.notes_to_monophonic_stretched_voice(notes, sample_rate=48000)
        audio = synthesizer.add_fx(
            audio, lowpass_filter_probability=self.lowpass_filter_probability,
            static_noise_amplitude=self.static_noise_amplitude,
            sample_rate=48000
        )
        # We re-build the segmentation from the notes
        descriptors = synthesizer.notes_to_descriptor(notes, sample_rate=44100)

        return audio, descriptors

    @staticmethod
    def _downscale_signal_to_given_length(signal, length, hop_length=441):
        re_sampled = [np.repeat(p, hop_length) for p in signal]
        re_sampled = np.concatenate(re_sampled)
        re_sampled = np.interp(
            np.linspace(0, len(re_sampled), length),  # Coordinates where to evaluate f
            np.arange(0, len(re_sampled)),            # x where f is known
            re_sampled                                # y = f(x)
        )
        return re_sampled


class DatasetReader(Dataset):
    """
    Load a precomputer dataset (collection of input_n.pt / output_n.pt)
    """

    def __init__(self, folder):
        """
        Initialize the reader from directory contents.

        :param folder: Folder containing the input/output files.
        """
        self.inputs = []
        self.outputs = []
        self.folder = folder
        if os.path.isdir(folder):
            self.length = len(glob.glob(folder + "/input_*.pt"))
        else:
            self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inp_path = self.folder + "/input_" + str(index) + ".pt"
        out_path = self.folder + "/output_" + str(index) + ".pt"

        # Check validity of the item
        if not os.path.isfile(inp_path):
            raise Exception(f"Missing file {inp_path}")
        if not os.path.isfile(out_path):
            raise Exception(f"Missing file {out_path}")
        if index >= self.length:
            raise IndexError

        # Load x,y pair
        inp = torch.load(inp_path)
        out = torch.load(out_path)
        return inp, out


def _audio_to_audio_and_descriptors(audio, sample_rate=48000, duration=None):
    """

    Parse an audio input and compute its

    :param audio: Audio input
    :param sample_rate: Sample rate of audio input
    :param duration: Time limitation expressed in seconds
    :param index: Give additional information for warning in case of silent sample.
    :return:
    """
    import v2p
    import scipy.signal as sps

    if sample_rate != 48000:
        raise "V2p's python binding only support sample_rate of 48000."

    # Convert time length to sample_length and truncate audio
    if duration:
        sample_length = int(duration * sample_rate)
        audio = audio[:sample_length]

    # Pitch expressed in midi numbers
    pitch = v2p.boersma_path(audio)
    numbers = v2p.pitch_to_midi_numbers(pitch)

    # Segmentation
    notes = v2p.midi_numbers_to_notes(numbers)
    width = np.zeros(len(pitch))  # silence (note = 0)
    offset = np.zeros(len(pitch))  # silence (note = 0)
    for note in notes:
        if note.note_number < 1:
            continue
        note_start = int(note.position + note.duration)
        note_end = int(note.position)
        interval_length = note_end - note_start
        width[note_start:note_end] = note.note_number
        offset[note_start:note_end] = np.arange(interval_length) / (interval_length - 1)

    # Resample to 44.1kHz
    # def rsz(signal):
    #     signal = [np.repeat(p, 44100 / 100) for p in signal]
    #     signal = np.concatenate(signal)
    #     return signal
    def rsz(signal):
        from scipy.interpolate import interp1d

        # Define the x values corresponding to the original signal
        x_original = np.linspace(0, len(signal) - 1, len(signal))

        # Define the x values for the interpolated signal
        x_interp = np.linspace(0, len(signal) - 1, len(signal) * int(44100 / 100))

        # Create a linear interpolation function using the original x and y values
        interp_func = interp1d(x_original, signal, kind='linear')

        # Evaluate the interpolation function at the new x values to generate the interpolated signal
        return interp_func(x_interp)

    numbers = rsz(numbers)
    width = rsz(width)
    offset = rsz(offset)

    # Resample audio signal to 44.1kHz
    audio = sps.resample_poly(audio, up=len(numbers), down=len(audio))

    assert len(audio) == len(numbers)
    assert len(numbers) == len(width)
    assert len(numbers) == len(offset)

    descriptors = np.array([width, offset, numbers])
    return audio, descriptors


def _signal_descriptors_to_frames_descriptors(signal, descriptors, frame_length=4096, hop_length=441):
    frames = signal_to_frames(signal=signal, frame_length=frame_length, hop_length=hop_length)
    descriptors = [downsample_signal_to_frame_scale(s, hop_length=441, frame_length=frame_length) for s in descriptors]
    descriptors = pitch_segmentation_to_segmentation_onehot(descriptors)
    return frames, descriptors


def _duration_uniformiser(self, signal, frames, descriptors, sample_rate=44100, hop_length=441):
    """
    Make the temporal dimension of `self` item equal, either by
    adding 0 at the end of the sample, or by truncating the sample.

    The duration (time limit) is taken from `self.duration`.
    If `duration` is None, then the current sample duration is used
    to set `self.duration`.

    :param self: A pitchnet dataset with a `.duration` member variable.
    :param signal: Input signal
    :param frames:
    :param descriptors:
    :return: (Signal, Frames, Descriptors) after processing.
    """
    if self.duration is None:
        self.duration = len(frames) * hop_length / sample_rate
        return signal, frames, descriptors

    audio_length = round(self.duration * sample_rate)
    info_length = round(self.duration * sample_rate / hop_length)
    if len(frames) < info_length:
        delta_audio = audio_length - len(signal)
        delta_info = info_length - len(frames)
        frames = np.pad(frames, ((0, delta_info), (0, 0), (0, 0)), mode='constant')
        descriptors = np.pad(descriptors, ((0, delta_info), (0, 0)), mode='constant')
        signal = np.pad(signal, (0, delta_audio), mode='constant')
    elif len(frames) > info_length:
        frames = frames[:info_length]
        descriptors = descriptors[:info_length]
        signal = signal[:audio_length]

    return signal, frames, descriptors
