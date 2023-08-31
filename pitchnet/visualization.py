import matplotlib.pyplot as plt
import numpy as np


def visualize_frames(frames, signal=None, min_norm_amplitude=0.01, apply_log=False, show_volume=True):
    """
    This function visualize a collection of processed frames. Each frame
    should have the format (Norm, Argument, Autocorrelation) where each element
    of the tuple is a `np.array`.

    :param signal: The input signal used to generate the frames.
    :param frames: The frames to be displayed (with format (Norm, Argument, Autocorrelation)). Shape is Tx3xF.
    :param min_norm_amplitude: The minimal amplitude for the norm of a fft bucket to appear in the graph.
    :param apply_log: Control whether the norm should be display after applying a logarithm (nicer display).
    :return:
    """

    # If called without signal, default it to 0
    if signal is None:
        signal = np.zeros(len(frames))
    signal = np.array(signal)
    frames = np.array(frames)

    # Reshape if wrong shape
    if frames.shape[1] != 3 and frames.shape[1] != 4:
        if frames.shape[0] == 3 or frames.shape[0] == 4:  # Its probably a 3xTxF, reshape to Tx3xF
            frames = frames.transpose(1, 0, 2)
        else:
            raise Exception("The shape of the frames should be Tx3xF.")

    # Select if we should display volume
    if show_volume and frames.shape[1] == 4:
        show_volume = True
    else:
        show_volume = False
    rows = 3 if show_volume else 2
    cols = 2

    # Start drawing the figure
    plt.figure()
    # --- Visualize norm
    p1 = plt.subplot(rows, cols, 1)
    p1.title.set_text('FT Norm')
    limy_value = max([np.nonzero(f[0] >= min_norm_amplitude)[0].max(initial=1) for f in frames] + [1])
    plt.ylim(top=limy_value)  # Limit the maximum y by `limy_value`
    norm = np.array([f[0] for f in frames]).T
    # Apply logarithm if required
    if apply_log:
        if norm.min() < 0:
            norm -= norm.min()
        norm = np.log(norm + 1e-12)
    p = plt.imshow(norm, aspect='auto', origin='lower')
    plt.colorbar(p, orientation="horizontal", pad=0.2)

    # --- Visualize phase
    p2 = plt.subplot(rows, cols, 2, sharex=p1, sharey=p1)
    p2.title.set_text('Phase')
    p = plt.imshow(np.array([f[1] for f in frames]).T, aspect='auto', origin='lower')
    plt.colorbar(p, orientation="horizontal", pad=0.2)

    # --- Visualize signal
    p3 = plt.subplot(rows, cols, 3, sharex=p1)
    p3.title.set_text('Pitch')
    plt.plot(np.linspace(0, len(frames), len(signal)), signal)

    # --- Visualize Autocorrelation
    p4 = plt.subplot(rows, cols, 4, sharex=p1, sharey=p1)
    p4.title.set_text('Autocorrelation')
    p = plt.imshow(np.array([f[2] for f in frames]).T, aspect='auto', origin='lower')
    plt.colorbar(p, orientation="horizontal", pad=0.2)

    # --- Visualize Volume
    if show_volume:
        p5 = plt.subplot(rows, cols, 5, sharex=p1)
        p5.title.set_text('Volume')
        p = plt.imshow(np.array([f[3] for f in frames]).T, aspect='auto', origin='lower')
        plt.colorbar(p, orientation="horizontal", pad=0.2)

    plt.subplots_adjust(
                # left=0.1,
                # bottom=0.1,
                # right=0.9,
                # top=0.9,
                # wspace=0.4,
                hspace=0.5,
    )
    plt.show()


def visualize_onehot_segmentation(monophonic_descriptor, convert_to_logprob=False):
    """

    Take a monophonic descriptor -- i.e. a np.array of shape Tx(1 + max_midi_numbers) = Tx129 -- and
    display the segmentation (first curve) and the pitch (remaining 128 curves).
    Each 1x128 slice is a softmax or LogSoftmax.

    The convert_to_logprob argument control whether the pitch input is supposed to be log probability (false) or
    probability (true).

    If false, input should be a log prob. If true, it is considered to be the output of a softmax.

    :param monophonic_descriptor: A monophonic descriptor (np.array of dim Tx(1+128))
    :param convert_to_logprob: Should we apply a logarithm to pitch input.
    :return:
    """
    pitch = np.array(monophonic_descriptor[:, -128:])
    segmentation_width = np.array(monophonic_descriptor[:, 0])
    segmentation_offset = np.array(monophonic_descriptor[:, 1])
    if monophonic_descriptor.shape[1] == 128 + 4:
        segmentation_confidence = np.array(monophonic_descriptor[:, 2])
        segmentation_presence = np.array(monophonic_descriptor[:, 3])
    else:
        segmentation_presence = np.array((monophonic_descriptor[:, 0] != 0) * 1.0)
        segmentation_confidence = segmentation_presence.copy()

    if segmentation_offset.min() < 0:
        segmentation_offset = segmentation_offset * 0.5 + 0.5

    if convert_to_logprob:
        pitch = np.log(pitch + 1e-12)

    plt.figure()
    ax1 = plt.subplot(122)
    color = 'tab:red'
    ax1.set_xlabel('time (samples)')
    ax1.set_ylabel('Note descriptor (in seconds)', color=color)
    ax1.plot(segmentation_width * (segmentation_presence > 0.5), color='blue')
    ax1.plot(segmentation_offset * (segmentation_presence > 0.5), color=color)
    ax1.plot(segmentation_confidence, color='green')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = plt.subplot(121, sharex=ax1)

    color = 'tab:blue'
    ax2.set_ylabel('Pitch (midi numbers)', color=color)
    ax2.imshow(pitch.T, aspect='auto', origin='lower')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
