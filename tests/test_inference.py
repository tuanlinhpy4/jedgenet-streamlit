import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from inference import CLASS_NAMES, JedgeNetPredictor, preprocess_image


ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = JedgeNetPredictor(
            ROOT / "weights" / "jedgenet_4class_seed5.pth"
        )

    def test_preprocess_shape_and_values(self) -> None:
        image = Image.new("RGB", (180, 120), color=(120, 60, 30))
        tensor = preprocess_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
        self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_checkpoint_predicts_four_scores(self) -> None:
        image = Image.open(ROOT / "assets" / "samples" / "cracked.png")
        prediction = self.predictor.predict(image)
        self.assertIn(prediction.class_name, CLASS_NAMES)
        self.assertEqual(len(prediction.scores), 4)
        self.assertAlmostEqual(sum(prediction.scores), 1.0, places=5)
        self.assertGreaterEqual(prediction.inference_ms, 0.0)

    def test_grayscale_input_is_supported(self) -> None:
        image = Image.new("L", (64, 64), color=128)
        prediction = self.predictor.predict(image)
        self.assertIn(prediction.class_name, CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
