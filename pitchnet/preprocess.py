import numpy as np
import scipy.signal
import librosa

from .tools import static_vars

__all__ = ['signal_to_frames', 'signal_to_sdft', 'sdft_to_frames',
           'compute_phase_vocoder', 'low_pass_signal',
           'pitch_to_onehot', 'onehot_to_pitch',
           'pitch_segmentation_to_segmentation_onehot',
           'downsample_signal_to_frame_scale']


def signal_to_frames(signal, frame_length=4096, hop_length=441):
    """
    Convert an audio signal  to a collection of 4-channels frames
    containing normalized amplitude, phase and autocorrelation.

    The signal is processed by a sliding window of size `frame_length`
    moved at each step by `hop_length`. The signal is padded with 0 on
    each side for computing the firsts and lasts frames.

    See `sdft_to_frames` for more information on which information
    are contained in each channel.

    :param signal: A numpy array of floats representing the audio signal
    :param frame_length: The length of the frame. Also the size of the FFT window.
    :param hop_length: The amount of samples to skip before cutting a new frame from the samples.
    :return:
    """
    sdft = signal_to_sdft(signal, frame_length=frame_length, hop_length=hop_length)
    frames_fft = sdft_to_frames(sdft, frame_length=frame_length, hop_length=hop_length)
    frame_volume = signal_to_volume_frame(signal=signal, frame_length=frame_length, hop_length=hop_length)
    frames = np.concatenate([frames_fft, frame_volume], axis=1)
    return frames


def signal_to_sdft(signal, frame_length=4096, hop_length=441):
    """
    Convert an signal to a collection of FFT frame.

    A frame is the real fourier transform of a window of size
    `window_size`. At each step, the window is slided by `step_size`
    samples on the right.

    If the audio signal is sampled at 44100 Hz, and the step_size
    is 411, the resulting frames are spaced in the temporal domain by
    10 ms.

    This function doesn't need to know the sampling frequency in order
    to compute the collection of frames.

    We use the scipy hann window for computing the FFT.

    The result is of shape T x S where:
      - T is the time index where one unit is hop_length/sample_rate seconds,
      - S is the frequency bin index between 0 and (frame_length / 2) + 1.

    :param signal: A numpy array of floats representing the audio signal
    :param frame_length: The size of the fft window
    :param hop_length: Number of samples between each frame
    :return: A TxS numpy array
    """

    # Apply a short time fourier transform.
    frames = librosa.core.stft(
        signal, n_fft=frame_length, win_length=frame_length,
        window=scipy.signal.windows.hann, hop_length=hop_length, pad_mode='constant'
    )  # Shape SxT
    frames = frames.swapaxes(0, 1)  # Shape TxS
    return frames


def downsample_signal_to_frame_scale(signal, frame_length=4096, hop_length=441):
    """

    Downsample a signal to match the scale of `signal_to_frames`.

    :param signal: The input collection of samples
    :param frame_length: Size of the fft window
    :param hop_length: Number of samples between each frame.
    :return: The subsampled signal.
    """
    signal = np.pad(signal, int(frame_length // 2), mode='constant')
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.array(frames[len(frames) // 2])


@static_vars(ac={})
def window_autocorrelation(window_size):
    """
    Return the autocorrelation of the hann window given its length.

    :param window_size: Length of the window
    :return: Autocorrelation of the window
    """
    if window_autocorrelation.ac.get(window_size, None) is None:
        v = np.fft.rfft(np.hanning(window_size))
        v = np.abs(v) ** 2
        v = np.fft.irfft(v)
        window_autocorrelation.ac[window_size] = v
    return window_autocorrelation.ac[window_size]


def sdft_to_frames(sdft, frame_length=4096, hop_length=441):
    """
    Split the fft frames in 3 channels:
      - magnitude of the fft bins, computed with np.abs
      - angle (rads) of the fft bins, computed with np.angle
      - the autocorrelation of the signal, corrected for windowing
      - the normalized volume
    It then apply the phase vocoder algorithm to remove the phase shift
    induced by the sliding window for the angle frame.

    :param sdft: The sliding discrete fourier transform as an np.array shaped TxS
    :param frame_length: Size of the window used while building frames from signal
    :param hop_length: The length of the step / hop between two frames
    :return: Processed frames as a list of np.array shaped Tx3xS
    """
    output = []

    # Compute amplitude / phase
    for f in sdft:
        # Compute
        autocorrelation = compute_corrected_ac(f, frame_length=frame_length)
        norm = np.abs(f)
        angle = np.angle(f)
        # Normalize
        norm -= norm.mean()
        norm_std = norm.std()
        if norm_std != 0:  # Prevent dividing by 0
            norm /= norm_std
        autocorrelation -= autocorrelation.mean()
        # Stack channels together
        f = np.array([norm, angle, autocorrelation])
        output += [f]
    output = np.array(output)

    # Compute angle with phase vocoder
    output[:, 1, :] = compute_phase_vocoder(output[:, 1, :], frame_length=frame_length, hop_length=hop_length)

    return output


def compute_corrected_ac(s, frame_length=4096):
    small = 1e-12
    # Compute AC
    ac = np.fft.irfft(np.abs(s) ** 2)
    ac = ac[:len(s)]

    # Correct AC by window's AC
    ac /= np.clip(window_autocorrelation(frame_length)[:len(s)], a_min=small, a_max=None)

    # Normalize and flip
    ac /= max(small, np.max(ac))
    ac = np.flip(ac, 0)

    return ac


def compute_phase_vocoder(frames, frame_length=4096, hop_length=441):
    """
    Take an array of frames and remove the phase delays from each frame.

    :param frames: A np.array shaped Tx1xS containing the time windowed FFT of the signal
    :param frame_length: The size of the FFT window
    :param hop_length: The step used by the slinding window of the SDFT
    :return: A np.array of same length as the input with smoothed phases
    """
    import math

    # Copy frame
    frames = frames.copy()

    # Cycle length of the sinusoids
    cycle_length = frame_length / np.arange(1, frame_length // 2)

    # For each frame, compute the phase shift
    for index, f in enumerate(frames):
        dphis = (2 * math.pi * index * hop_length) / cycle_length
        # Remove phase shift from the frame
        frame_slice = frames[index, 1:frame_length//2]
        frames[index, 1:frame_length//2] = np.mod(frame_slice - dphis, 2 * math.pi)
    return frames


# TODO: Test if we really need to low pass the signal prior to training the network.
def low_pass_signal(signal):
    """
    Apply gaussian filter to a signal in order to lessen noise.

    :param signal: Input signal
    :return: Filtered signal
    """
    import scipy.ndimage

    signal = scipy.ndimage.gaussian_filter(signal, 4)
    return signal


def signal_to_volume_frame(signal, frame_length=4096, hop_length=441):
    # Compute local max on a frame_length // 2 window
    from scipy.ndimage.filters import maximum_filter1d

    kernel_size = frame_length // 2
    volume = maximum_filter1d(np.abs(signal), kernel_size, mode='constant')

    assert len(volume) == len(signal)
    volume = downsample_signal_to_frame_scale(volume, frame_length=frame_length, hop_length=hop_length)
    frames = np.tile(volume[:, None, None], (1, 1, frame_length // 2 + 1))

    # Normalize
    frames -= frames.mean()
    std = frames.std()
    frames /= std if std > 0 else 1
    return frames


def pitch_to_onehot(midi_number: float, vector_length=128):
    """
    Convert a pitch expressed in midi numbers to a probability distribution
    over a 128 keys vector, where each cell represent a keyboard key / midi number.

    For example, the midi_number 2 will be translated to the
    one hot vector [0, 0, 1, 0, ... 0].

    A midi_number of 1.5 will be translated to the probability distribution
    [0, 0.5, 0.5, 0, ... 0].

    :param midi_number: A midi number (music pitch).
    :param vector_length: The number of keys.
    :return:
    """
    out = np.zeros(vector_length)

    # Skip invalid midi_number
    if midi_number >= vector_length or midi_number < 0:
        raise Exception(f"Invalid midi_number {midi_number} outside of the interval [0, {vector_length})")

    loc = int(midi_number)
    val = midi_number - loc

    out[loc] = 1 - val
    if loc + 1 < vector_length:
        out[loc + 1] = val

    return out


def onehot_to_pitch(vector):
    """
    This function allow to convert a probability (or log probability)
    distribution over the midi number to an actual midi number value.

    This function select the two maximal positives values and
    compute the weighted mean of indices.

    :param vector: Probability distribution over midi numbers.
    :return:
    """
    max_args = np.argsort(vector)[-2:]
    val_args = vector[max_args].clip(0, None)
    num_args = np.arange(len(vector))[max_args]

    # In case there is no values
    if val_args.sum() == 0:
        return 0

    return (num_args * val_args).sum() / val_args.sum()


def onehot_to_pitch_torch(vector):
    """
    This function allow to convert a probability (or log probability)
    distribution over the midi number to an actual midi number value.

    This function select the two maximal positives values and
    compute the weighted mean of indices.

    :param vector: Probability distribution over midi numbers. Shape NxTxP
    :return:
    """
    import torch
    val_arg1, num_arg1 = torch.max(vector, dim=-1, keepdim=True)

    cpy = vector.clone()

    # ---
    x = cpy
    # get the indices of the maximum values along the last dimension
    _, indices = torch.max(x, dim=-1)

    # create a range of indices for the first and second dimensions
    indices_1 = torch.arange(x.shape[0]).unsqueeze(-1).repeat(1, x.shape[1]).view(-1)
    indices_2 = torch.arange(x.shape[1]).repeat(x.shape[0])

    # flatten the tensor and the indices
    x_flat = x.view(-1, x.shape[-1])
    indices_flat = indices.view(-1)

    # use the indices to set the corresponding values to zero
    x_flat[indices_1, indices_2, indices_flat] = 0

    # reshape the tensor back to its original shape
    x = x_flat.view(x.shape)

    # Do it again
    val_arg2, num_arg2 = torch.max(x, dim=-1, keepdim=True)

    return (num_arg1 * val_arg1 + num_arg2 * val_arg2) / (val_arg1 + val_arg2)


def onehot_to_pitch_torch3(val_args):
    """
    This function allow to convert a probability (or log probability)
    distribution over the midi number to an actual midi number value.

    This function select the two maximal positives values and
    compute the weighted mean of indices.

    TODO: Move to tools file probably
    TODO: Add asserts on non negative

    :param vector: Probability distribution over midi numbers. Shape NxTxP
    :return:
    """
    x = np.arange(val_args.shape[-1])
    x = x.reshape(1, 1, -1)
    num_args = np.tile(x, (val_args.shape[0], val_args.shape[1], 1))
    import torch
    r = (torch.Tensor(num_args).to(val_args.device) * val_args).sum(dim=-1) / val_args.sum(dim=-1)
    return r

def onehot_to_pitch_torch2(vector):
    """
    This function allows to convert a probability (or log probability)
    distribution over the midi number to an actual midi number value.

    This function selects the two maximal positive values and
    computes the weighted mean of indices.

    :param vector: Probability distribution over midi numbers. Shape NxTxP
    :return:
    """
    import torch

    # Step 1: Find the top two maximum values and their indices
    val_args, max_args = torch.topk(vector, 2, dim=-1)

    # Step 2: Clip these values to be within the range [0, inf]
    val_args = torch.clamp_min(val_args, 0)

    # Create an indices tensor that matches the size of the last dimension of the input tensor
    indices = torch.arange(vector.shape[-1]).expand(vector.shape[:-1] + (-1,)).to(vector.device)

    # Get the indices of the top two maximum values
    num_args = indices.gather(-1, max_args)

    # Step 3: Compute the weighted average of the indices
    weighted_avg = torch.sum(num_args * val_args, dim=-1) / torch.sum(val_args, dim=-1)

    # Step 4: If sum of the values is 0, return 0; else return weighted_avg
    result = torch.where(torch.sum(val_args, dim=-1) == 0, torch.zeros_like(weighted_avg), weighted_avg)

    return result

def pitch_segmentation_to_segmentation_onehot(descriptors):
    """
    Return width [1], offset [1], and midi notes [128] in a single np.array.
    """
    seg_desc_w, seg_desc_o, pitch_desc = descriptors
    onehots = np.array([pitch_to_onehot(v) for v in pitch_desc])
    return np.concatenate([seg_desc_w[:, None], seg_desc_o[:, None], onehots], axis=1)
