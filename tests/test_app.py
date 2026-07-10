import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_initial_and_sample_states_render_without_errors(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.segmented_control[0].value, "Upload")
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.button), 4)

        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        rendered = "\n".join(block.value for block in app.markdown)
        self.assertIn("Predicted class", rendered)
        self.assertIn("Class scores", rendered)

    def test_camera_state_renders_without_errors(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.segmented_control[0].set_value("Camera").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.file_uploader), 0)


if __name__ == "__main__":
    unittest.main()
