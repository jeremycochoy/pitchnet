import os
import numpy as np
from datetime import datetime

__all__ = ['wav_load', 'wav_save', 'export_dataset', 'export_wav', 'import_dataset']


def wav_load(path, sample_rate=44100, duration=None):
    """
    Load a wav file in memory given a path, and re-sample it to `sample_rate`
    in mono.
    One can limit the duration of the file loaded with `duration`.
    The argument `duration` is expressed in seconds.

    For stereo wav file, only the first (left) channel is used.

    :param path: File path
    :param sample_rate: Sampling rate of the audio returned. Default is 44100.
    :param duration: Limit the duration of the audio loaded from the file, expressed in seconds.
                          Default to None = No limit.
    :return: A np.array of samples.
    """
    from scipy.io import wavfile
    import scipy.signal as sps
    file_sampling_rate, data = wavfile.read(path)

    # Remove stereo
    if len(data.shape) > 1:
        # Take left channel
        data = data.T[0]

    # Limit length
    if duration:
        data = data[:duration * file_sampling_rate]

    # Resample data if required
    if file_sampling_rate != sample_rate:
        old_len = len(data)
        new_len = int(old_len * sample_rate / file_sampling_rate)
        return sps.resample_poly(1.0 * data, new_len, old_len), sample_rate
    else:
        return data, file_sampling_rate


def wav_save(filename, sample_rate, samples):
    """
    Save an audio signal into a wav file.

    :param filename: File to write to
    :param samples: A numpy array of samples
    :param sample_rate: The sample rate of the audio signal
    :return:
    """
    from scipy.io import wavfile

    wavfile.write(filename=filename, rate=sample_rate, data=samples)


def export_dataset(dataset, folder=None, verbose=False, skip_exists=False):
    """
    This function export a dataset as a folder of pytorch tensor (.pt) files.
    This allow precompute of dataset for faster training.

    This will create a folder in the current working directory.

    :param dataset: A Pitchnet Dataset
    :param folder: Name of the export folder
    :param verbose: Display log message for each exported file
    :param skip_exists: Should we ignore existing file or replace them ?
    :return:
    """
    import torch

    if folder is None:
        folder = type(dataset).__name__

    if not os.path.isdir(folder):
        if os.path.isfile(folder):
            print("Error: " + folder + " is not a valid directory")
            return
        os.mkdir(folder)
    folder += "/"

    for i in range(len(dataset)):
        ifile = folder + "input_" + str(i) + ".pt"
        ofile = folder + "output_" + str(i) + ".pt"

        if skip_exists:
            if os.path.isfile(ifile) and os.path.isfile(ofile):
                if verbose:
                    print("Skip item %d" % i)
                continue

        inp, out = dataset[i]
        torch.save(inp, ifile)
        torch.save(out, ofile)
        if verbose:
            print(f"{datetime.now()} - Export item %d" % i)


def import_dataset(folder):
    from .dataset import DatasetReader

    return DatasetReader(folder=folder)

def export_wav(dataset, folder=None):
    """
    This function export a wav file from a dataset providing
    the method `__getwav__`.

    :param dataset: A Pitchnet Dataset
    :param folder: Name of the export folder
    :return:
    """
    # Get function
    wavfunc = getattr(dataset, "__getwav__", None)
    if not callable(wavfunc):
        raise Exception("This dataset do not provide __getwav__.")

    # Get folder
    if folder is None:
        folder = type(dataset).__name__

    if not os.path.isdir(folder):
        if os.path.isfile(folder):
            print("Error: " + folder + " is not a valid directory")
            return
        os.mkdir(folder)

    # Export audio
    for i in range(len(dataset)):
        signal, sample_rate = dataset.__getwav__(i)
        signal = np.array(signal)
        filename = folder + "/" + str(i) + ".wav"
        wav_save(filename=filename, sample_rate=sample_rate, samples=signal)
