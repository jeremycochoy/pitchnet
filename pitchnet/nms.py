import torch
from pitchnet.loss import split_in_seg_pitch_prediction
from pitchnet.preprocess import onehot_to_pitch_torch2
from pitchnet.tools import calculate_iou, calculate_iou_box
from pitchnet.synthesis import Note
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def nms(x, iou_threshold=0.5, sampling_interval=0.01, presence_threshold=0.5, tolerance=0.5):
    """
    Apply Non-Maximal Suppression to keep only the most relevant notes.

    :param x: Tensor or array of shape Tx(4+128), containing segmentation and pitch information.
    :param iou_threshold: IoU threshold for suppression. Notes with IoU greater than this threshold will be suppressed.
    :param sampling_interval: Sampling interval in seconds. Default is 10ms.
    :return: List of selected Note objects.
    """

    # Detect the library based on the input type
    library = torch if isinstance(x, torch.Tensor) else np

    # Add a dummy batch dimension to x
    x = x.unsqueeze(0) if library == torch else np.expand_dims(x, 0)

    # Step 1: Extract variables
    x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence, x_pitch = split_in_seg_pitch_prediction(x)
    x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence = x_seg_width.squeeze(0), x_seg_offset.squeeze(0), x_seg_confidence.squeeze(0), x_seg_presence.squeeze(0)

    # Step 2: Calculate pitch
    if (x_pitch < 0).any(): # We probably received logProbabilities by mistake
        pitch = onehot_to_pitch_torch2(torch.exp(x_pitch)).squeeze(0)
    else:
        pitch = onehot_to_pitch_torch2(x_pitch).squeeze(0)
    print(pitch)

    # Step 3: Calculate objectiveness scores
    scores = x_seg_presence * x_seg_confidence

    # Step 4: Sort notes by score
    sorted_indices = library.argsort(scores, descending=True)

    # Step 5: Initialize selected notes list
    selected_notes = []

    # Step 6: Iterate through sorted notes
    for i in sorted_indices:
        # Skip zero notes
        if x_seg_presence[i].item() < presence_threshold or pitch[i] < 1:
            continue

        note_candidate = Note()
        note_candidate.number = pitch[i].item()
        note_candidate.position = i.item() * sampling_interval + x_seg_offset[i].item() * x_seg_width[i].item() / 2
        note_candidate.duration = x_seg_width[i].item()
        note_candidate.confidence = scores[i].item()
        print(note_candidate)

        suppress = False
        for selected_note in selected_notes:
            iou = calculate_iou_box(
                w1=library.tensor([note_candidate.duration]),
                h1=library.tensor(tolerance),
                x1=library.tensor([note_candidate.position]),
                y1=library.tensor([note_candidate.number]),

                w2=library.tensor([selected_note.duration]),
                h2=library.tensor(tolerance),
                x2=library.tensor([selected_note.position]),
                y2=library.tensor([selected_note.number]),
            )
            if iou > iou_threshold:
                suppress = True
                break

        if not suppress:
            selected_notes.append(note_candidate)

    # Step 7: Return selected notes
    return selected_notes


def plot_notes(notes):
    """
    Plot a list of notes on a pyplot graph.

    :param notes: List of Note objects.
    :return: None
    """

    # Create a colormap for confidence scores
    cmap = plt.get_cmap('coolwarm_r')

    # Create a normalizer for confidence scores
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the y-axis limits to match the MIDI note range
    ax.set_ylim([0, 128])
    ax.set_xlim([0, 7])

    # Set the x-axis label
    ax.set_xlabel('Time (s)')

    # Plot each note
    for note in notes:
        ax.add_patch(plt.Rectangle((note.position, note.number), note.duration, 1,
                                   color=cmap(norm(note.confidence)), alpha=0.5))

    # Set the y-axis label
    ax.set_ylabel('Pitch (MIDI)')

    # Add a colorbar
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Confidence')

    # Show the plot
    plt.show()


def all_notes(x, sampling_interval=0.01, presence_threshold=0.5):
    """
    """

    # Detect the library based on the input type
    library = torch if isinstance(x, torch.Tensor) else np

    # Add a dummy batch dimension to x
    x = x.unsqueeze(0) if library == torch else np.expand_dims(x, 0)

    # Step 1: Extract variables
    x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence, x_pitch = split_in_seg_pitch_prediction(x)
    x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence = x_seg_width.squeeze(0), x_seg_offset.squeeze(0), x_seg_confidence.squeeze(0), x_seg_presence.squeeze(0)

    # Step 2: Calculate pitch
    if (x_pitch < 0).any(): # We probably received logProbabilities by mistake
        pitch = onehot_to_pitch_torch2(torch.exp(x_pitch)).squeeze(0)
    else:
        pitch = onehot_to_pitch_torch2(x_pitch).squeeze(0)

    # Step 3: Build a list of notes
    notes = []
    for i in range(len(pitch)):
        if x_seg_presence[i] <= presence_threshold or pitch[i] < 1:
            continue
        note = Note()
        note.number = pitch[i].item()
        note.position = i * sampling_interval + x_seg_offset[i].item() * x_seg_width[i].item() / 2 /10
        note.duration = x_seg_width[i].item()
        note.confidence = x_seg_presence[i].item()
        notes.append(note)
    return notes

def hds(x, iou_threshold=0.5, sampling_interval=0.01, presence_threshold=0.5, tolerance=0.5):
    """
    Apply the High Density Selection (HDS) algorithm to select the most relevant notes.

    :param x: Tensor or array of shape Tx(4+128), containing segmentation and pitch information.
    :param iou_threshold: IoU threshold for selection. Notes with IoU greater than this threshold will be considered.
    :param sampling_interval: Sampling interval in seconds. Default is 10ms.
    :return: List of selected Note objects.
    """

    # Detect the library based on the input type
    library = torch if isinstance(x, torch.Tensor) else np

    # Add a dummy batch dimension to x
    x = x.unsqueeze(0) if library == torch else np.expand_dims(x, 0)

    # Step 1: Extract variables
    x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence, x_pitch = split_in_seg_pitch_prediction(x)
    x_seg_width, x_seg_offset, x_seg_confidence, x_seg_presence = x_seg_width.squeeze(0), x_seg_offset.squeeze(0), x_seg_confidence.squeeze(0), x_seg_presence.squeeze(0)

    # Step 2: Calculate pitch
    if (x_pitch < 0).any(): # We probably received logProbabilities by mistake
        pitch = onehot_to_pitch_torch2(torch.exp(x_pitch)).squeeze(0)
    else:
        pitch = onehot_to_pitch_torch2(x_pitch).squeeze(0)

    # TODO: Can be speed up by note creating notes when presence is lower than presence_threshold

    # Step 3: Build a list of notes
    notes = []
    for i in range(len(pitch)):
        if x_seg_presence[i] < presence_threshold or pitch[i] < 1:
            continue
        note = Note()
        note.number = pitch[i].item()
        note.position = i * sampling_interval + x_seg_offset[i].item() * x_seg_width[i].item() / 2
        note.duration = x_seg_width[i].item()
        note.confidence = x_seg_presence[i].item()
        notes.append(note)
    print("DEBUG notes len:", len(notes))

    # Step 4: Compute the IoU matrix
    iou_matrix = calculate_iou_matrix(notes, library, tolerance)
    print("IOU computed")

    # Step 5: Compute scores as the sum of IoUs
    scores = iou_matrix.sum(axis=-1)
    print("IOU scores:", scores)

    # Step 6: Select notes based on scores
    selected_notes = []
    while scores.max() > 0:
        max_score_idx = scores.argmax()
        selected_note = notes[max_score_idx]
        if selected_note.confidence < presence_threshold or selected_note.number < 1:
            scores[max_score_idx] = 0
            continue
        selected_notes.append(selected_note)
        suppress_indices = np.where(iou_matrix[max_score_idx] > iou_threshold)[0]
        scores[suppress_indices] = 0
        scores[max_score_idx] = 0

        print("Add note:", selected_note)

    # Step 7: Return selected notes
    return selected_notes

def calculate_iou_matrix(notes, library, tolerance=0.5):
    """
    Calculate the IoU matrix for a list of notes.

    :param notes: List of Note objects.
    :param library: Numpy or PyTorch.
    :param x_seg_offset: Segment offset values.
    :param tolerance: Tolerance value for IoU calculation.
    :return: IoU matrix.
    """

    num_notes = len(notes)
    iou_matrix = library.zeros((num_notes, num_notes))
    for i in range(num_notes):
        for j in range(i+1, num_notes):
            iou = calculate_iou_box(
                w1=library.tensor([notes[i].duration]),
                h1=library.tensor(tolerance),
                x1=library.tensor([notes[i].position]),
                y1=library.tensor([notes[i].number]),

                w2=library.tensor([notes[j].duration]),
                h2=library.tensor(tolerance),
                x2=library.tensor([notes[j].position]),
                y2=library.tensor([notes[j].number]),
            )
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    return iou_matrix
