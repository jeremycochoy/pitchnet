import numpy as np
from pitchnet.preprocess import onehot_to_pitch


def segmentation_to_notes_indices(segmentation):
    """
    Return a list of pairs of start & end of the notes described by a segmentation.

    The output is of the form `[[start: int, end: int]]`.
    """
    notes = []
    current_indicator = segmentation[0]
    start = 0
    for i in range(len(segmentation)):
        # Still browsing the same note/silence
        if segmentation[i] == current_indicator:
            continue

        if current_indicator == 0:
            # We was browsing a silence
            start = i
            current_indicator = segmentation[i]
        else:
            # We are browsing a note
            end = i - 1
            notes.append([start, end])
            # Now start a new note/silence
            start = i
            current_indicator = segmentation[i]
    return notes


def notes_indices_to_midi(pitch, notes):
    """
    Take a list of paires (start, end) and convert them to a list of triplets
    (midi_number, position, duration)
    """
    pitch = np.array(pitch)
    midi = []
    for note in notes:
        start, end = note
        pitch_slide = pitch[start:end]
        midi_number = np.median(pitch_slide[pitch_slide > 1])
        duration = end - start
        midi += [(midi_number, start, duration)]
    return midi


def compute_midi_from_descriptors(descriptors):
    y = np.array(descriptors)

    # Segmentation and Pitch curve
    segmentation = y[:, 0]  # shape = T
    pitch = y[:, 1:]  # shape = Tx128
    pitch = np.array(list(map(onehot_to_pitch, pitch)))  # shape = T

    # Build the notes (start, end) indes
    notes = segmentation_to_notes_indices(segmentation)
    notes = notes_indices_to_midi(pitch, notes)
    return notes
