import torch

from pitchnet.preprocess import onehot_to_pitch_torch2, onehot_to_pitch
from pitchnet.tools import calculate_iou

def split_in_seg_pitch_prediction(vector):
    """
    Split a prediction / target into segmentation
    curve and pitch one hot.

    Prediction/Target are shaped NxTx(2 + 128)
    Output is shaped (NxT, NxT, NxT, NxTx128) with in order width, offset, confidence and pitch.
    """
    w, o, c, p, pitch = vector[:, :, 0], vector[:, :, 1], vector[:, :, 2], vector[:, :, 3], vector[:, :, -128:]
    return w, o, c, p, pitch


def split_in_seg_pitch_reference(vector):
    """
    Split a prediction / target into segmentation
    curve and pitch one hot.

    Prediction/Target are shaped NxTx(2 + 128)
    Output is shaped (w: NxT, o: NxT, p: NxT, c: NxT, pitch: NxTx128) for respectively
        in order width, offset, presence, confidence and pitch.
    """
    w, o, pitch = vector[:, :, 0], vector[:, :, 1], vector[:, :, -128:]
    p = (w != 0) * 1.0
    c = p.clone()
    return w, o, c, p, pitch


def loss_fn(prediction, target, do_abs=False, segmentation_only=False, gamma=0.1):
    """
    This loss is used for computing both segmentation and pitch,
    using synthetic data from synthetizers and known pitch.
    """
    assert target.shape[-1] == 128 + 2
    seg2_width, seg2_offset, _, p2, pitch2 = split_in_seg_pitch_reference(target)

    if prediction.shape[-1] == 128 + 2:
        seg1_width, seg1_offset, c1, p1, pitch1 = split_in_seg_pitch_reference(prediction)
    else:
        assert prediction.shape[-1] == 128 + 4
        seg1_width, seg1_offset, c1, p1, pitch1 = split_in_seg_pitch_prediction(prediction)

    enable_presence = False
    enable_offset = False
    enable_width = False
    enable_confidence = False
    enable_iou = False
    enable_onset = False

    # --- PITCH

    # We ignore pitch from prediction if target pitch is zero.
    # We do that by replacing the prediction in this areas by zeros.
    z_tmp = torch.zeros(tuple(pitch1.shape)).to(pitch1.device)  # + y1.min()
    p = pitch2[:, :, 1:].sum(dim=2) > 0  # Collaps pitch dimension
    z_tmp[p] = pitch1[p]  # Copy y1 values where y2 is a non zero pitch
    pitch1 = z_tmp

    # Compute cost for pitch
    z1 = pitch1.reshape(-1, 128)
    z2 = pitch2.reshape(-1, 128)
    pitch_loss = torch.nn.functional.kl_div(z1, z2, reduction='batchmean')

    # --- Segmentation

    # In case no segmentation is available, do not compute the loss
    has_segmentation = (seg2_width.sum(axis=-1).sum(axis=-1) > 0) * 1.0  # NxT
    has_pitch = (pitch2[:, :, 1:].sum(dim=2) != 0) * 1.0 # NxT

    if enable_width:
        width_loss = torch.nn.functional.mse_loss(seg1_width ** 0.25 * (seg2_width != 0), seg2_width ** 0.25)
        # segmentation_width_loss = torch.nn.functional.mse_loss(seg1_width ** 0.5 * (seg2_width != 0), seg2_width ** 0.5)
        # segmentation_width_loss = torch.nn.functional.mse_loss(seg1_width * (seg2_width != 0), seg2_width)
    else:
        width_loss = torch.zeros_like(pitch_loss)

    if enable_offset:
        offset_loss = torch.nn.functional.mse_loss(seg1_offset * (seg2_offset != 0),
                                                            (seg2_offset * 2 - 1) * (seg2_offset != 0))
    else:
        offset_loss = torch.zeros_like(pitch_loss)

    if enable_presence:
        presence_loss = torch.nn.functional.mse_loss(p1 * has_segmentation, p2 * has_segmentation)
        # presence_loss += 0.5 * torch.nn.functional.mse_loss(p1*has_pitch, has_pitch)  # Voice audio doesn't have segmentation so we try to make for it
    else:
        presence_loss = torch.zeros_like(pitch_loss)

    if enable_confidence:
        confidence_loss = torch.nn.functional.mse_loss(c1 * p2, iou.detach() * p2)
    else:
        confidence_loss = torch.zeros_like(pitch_loss)

    if enable_onset:
        # Onset
        if p2.sum() <= 0:
            onset_loss = 0
        else:
            onset_loss = torch.nn.functional.mse_loss(seg1_offset * seg1_width / 2 * p2,
                                                      (seg2_offset * 2 - 1) * seg2_width / 2 * p2) * p2.shape[0] * p2.shape[
                             1] / p2.sum()
            print("DEBUG ONSET", f"min:{onset_loss.min().item()} < mean:{onset_loss.mean()} median:{onset_loss.median()} < max:{onset_loss.max().item()}")
    else:
        onset_loss = torch.zeros_like(pitch_loss)

    # Compute the IOU and pitch delta
    c_pitch1 = onehot_to_pitch_torch2(torch.exp(pitch1)) * p2
    c_pitch2 = onehot_to_pitch_torch2(pitch2) * p2
    if enable_iou:
        iou = calculate_iou(seg1_width,seg1_offset, c_pitch1, seg2_width, seg2_offset * 2 - 1, c_pitch2)
        iou_loss = ((iou * p2 - p2)**2).sum() / p2.sum() #MSE only on non zero notes
        print("DEBUG IOU", f"min:{iou.min().item()} < mean:{iou.mean()} median:{iou.median()} < max:{iou.max().item()}")
    else:
        iou_loss = torch.zeros_like(pitch_loss)
    deltap = (c_pitch1 - c_pitch2).abs()
    print(f"DEBUG PITCH reconstructed\tmin:{deltap.min().item()} < mean:{deltap.mean()} median:{deltap.median()} < max:{deltap.max().item()}")

    loss = pitch_loss
    if enable_offset:
        loss += offset_loss
    if enable_presence:
        loss += presence_loss
    if enable_width:
        loss += width_loss * 30
    if enable_onset:
        loss += onset_loss
    if enable_confidence:
        loss == confidence_loss

    return loss, pitch_loss, width_loss, offset_loss, presence_loss, confidence_loss, iou_loss, onset_loss
