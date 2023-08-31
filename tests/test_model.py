import unittest
import torch

from pitchnet.model import build_model, build_segmentation_model
from pitchnet.dataset import MonophonicDataset


class TestModelShapes(unittest.TestCase):

    def test_dataset_shapes_match_model_io_shapes(self):
        # Build a dataset just for dummy input shapes
        training_set = MonophonicDataset(notes_per_sample=1, size=1)
        x, y = training_set.__getitem__(0)

        # Speed up test by reducing the time dimension
        x, y = x[:10], y[:10]

        # Input shape
        inshape = x.shape
        # Output shape
        outshape = y.shape

        # Create a dummy input
        dummy_input = torch.ones(inshape)
        # Create model
        model = build_model()

        # Run through model (check the inshape match model's input)
        output = model(dummy_input[None])

        # Check output shape match label shape
        self.assertEqual(list(output.shape[1:]), list(outshape))

        # Create a segmentation model
        segmentation_model = build_segmentation_model(model)

        # Run the model and check output shape
        output = segmentation_model(dummy_input[None])
        self.assertEqual(list(output.shape[-2:]), [
            segmentation_model[-1].nb_midi_numbers, segmentation_model[-1].nb_coordinates
        ])


if __name__ == '__main__':
    unittest.main()
