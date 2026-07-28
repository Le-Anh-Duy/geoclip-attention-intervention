import unittest

import torch

from app.intervention import InterventionState


class InterventionStateTest(unittest.TestCase):
    def test_builds_pre_softmax_bias_and_keeps_cls_neutral(self):
        state = InterventionState(layer_ab={3: (5.0, -2.0)}, in_region_mask=torch.tensor([True, False]))

        bias = state.key_bias_for_layer(3, 3)

        torch.testing.assert_close(bias, torch.tensor([0.0, 5.0, -2.0]))
        self.assertIsNone(state.key_bias_for_layer(2, 3))


if __name__ == "__main__":
    unittest.main()
