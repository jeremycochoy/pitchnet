import unittest

from pitchnet.tools import calculate_iou

class TestCalculateIoU(unittest.TestCase):
    def test_disjoint_boxes(self):
        # These boxes do not intersect
        width1, offset1, pitch1 = 4, 1, 1
        width2, offset2, pitch2 = 3, 5, 5

        result_iou = calculate_iou(width1, offset1, pitch1, width2, offset2, pitch2, tolerance=0.5)

        # The IoU for disjoint boxes should be 0
        self.assertEqual(result_iou, 0.0)

    def test_intersecting_boxes(self):
        # These boxes intersect
        width1, offset1, pitch1 = 4, 5, 5
        width2, offset2, pitch2 = 3, 4, 4

        result_iou = calculate_iou(width1, offset1, pitch1, width2, offset2, pitch2, tolerance=3)

        # The IoU for these boxes should be > 0
        self.assertGreater(result_iou, 0)

        # The IoU for these boxes should be < 1 (not equal)
        self.assertLess(result_iou, 1)

# Run the tests
if __name__ == "__main__":
    unittest.main()
