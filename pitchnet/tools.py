__all__ = ['static_vars', 'calculate_iou', 'calculate_iou_box']

import torch
import numpy as np

def static_vars(**kwargs):
    """
    Add static variables to a function as a dictionary.

    One function can access it's static variables by
    acessing the dictionary `self.ac`.

    :param kwargs: Function's arguments
    :return:
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

def calculate_iou(width1, offset1, pitch1, width2, offset2, pitch2, tolerance=0.5):
    """
    assume offset in [-1,1]
    """

    # Select which library to use for computing the maximum
    if isinstance(width1, torch.Tensor):
        library = torch
    else:
        library = np

    # Create tolerance tensor/array with the same shape as width1 and width2
    tolerance = library.full_like(width1, fill_value=tolerance)

    # Deconstruct boxes into dimensions and center
    w1, h1, x1, y1 = width1, tolerance, width1 * offset1 / 2, pitch1
    w2, h2, x2, y2 = width2, tolerance, width2 * offset2 / 2, pitch2

    return calculate_iou_box(w1, h1, x1, y1, w2, h2, x2, y2)

def calculate_iou_box(w1, h1, x1, y1, w2, h2, x2, y2):
    """
    # Each box is a list/tuple with [width, height, x_center, y_center]
    """

    # Select which library to use for computing the maximum
    if isinstance(w1, torch.Tensor):
        library = torch
    else:
        library = np

    # Calculate box corners for the first box
    box1_x1 = x1 - w1 / 2
    box1_y1 = y1 - h1 / 2
    box1_x2 = x1 + w1 / 2
    box1_y2 = y1 + h1 / 2

    # Calculate box corners for the second box
    box2_x1 = x2 - w2 / 2
    box2_y1 = y2 - h2 / 2
    box2_x2 = x2 + w2 / 2
    box2_y2 = y2 + h2 / 2

    # Determine the coordinates of the intersection rectangle
    x_left = library.maximum(box1_x1, box2_x1)
    y_top = library.maximum(box1_y1, box2_y1)
    x_right = library.minimum(box1_x2, box2_x2)
    y_bottom = library.minimum(box1_y2, box2_y2)

    # Initialize intersection area as zero
    intersection_area = library.zeros_like(x_left)

    # Calculate intersection area only for boxes that do intersect
    intersect_mask = library.logical_and(x_right >= x_left, y_bottom >= y_top)
    intersection_area[intersect_mask] = (x_right[intersect_mask] - x_left[intersect_mask]) * (
                y_bottom[intersect_mask] - y_top[intersect_mask])

    # Calculate the area of both rectangles
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Calculate the union of both rectangles
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    eps = library.finfo(library.float32).eps if library == np else torch.finfo(torch.float32).eps
    union_area = library.where(union_area == 0, union_area + eps, union_area)

    # Compute IoU
    iou = intersection_area / union_area

    return iou
