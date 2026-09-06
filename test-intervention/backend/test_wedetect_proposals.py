import unittest

import numpy as np
from PIL import Image

from app.wedetect_proposals import WeDetectUniONNX, _nms, letterbox, rescale_boxes


class FakeSession:
    def run(self, output_names, inputs):
        self.output_names = output_names
        self.input_shape = inputs["input_image"].shape
        scores = np.zeros((1, 3, 256), dtype=np.float32)
        scores[0, 0, 4] = 0.9
        scores[0, 1, 7] = 0.8
        scores[0, 2, 2] = 0.05
        boxes = np.array(
            [[[64, 192, 320, 320], [70, 196, 315, 318], [400, 400, 500, 500]]],
            dtype=np.float32,
        )
        return scores, boxes


class WeDetectProposalTest(unittest.TestCase):
    def test_letterbox_and_rescale_round_trip(self):
        image = Image.new("RGB", (100, 50))
        padded, ratio, pad = letterbox(image, 640)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual(ratio, 6.4)
        np.testing.assert_allclose(pad, (0, 160))

        restored = rescale_boxes(
            np.array([[64, 192, 320, 320]], dtype=np.float32), ratio, pad, 100, 50
        )
        np.testing.assert_allclose(restored, [[10, 5, 50, 25]])

    def test_nms_keeps_highest_scoring_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        np.testing.assert_array_equal(_nms(boxes, scores, 0.5, 10), [0, 2])

    def test_generate_uses_max_internal_prompt_and_rescales(self):
        proposer = WeDetectUniONNX(model_path="unused-in-test")
        proposer._session = FakeSession()

        proposals = proposer.generate(Image.new("RGB", (100, 50)), score_threshold=0.1, top_k=10)

        self.assertEqual(len(proposals), 1)
        self.assertAlmostEqual(proposals[0].score, 0.9)
        self.assertAlmostEqual(proposals[0].x1, 10)
        self.assertAlmostEqual(proposals[0].y1, 5)
        self.assertEqual(proposer._session.input_shape, (1, 3, 640, 640))
        self.assertEqual(proposer._session.output_names, ["scores", "bboxes"])


if __name__ == "__main__":
    unittest.main()
