import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_builtin_data_contains_complete_test_split(self) -> None:
        expected_counts = {
            "cracked": 93,
            "dry": 221,
            "insect_damaged": 130,
            "invalid": 75,
        }
        test_data_dir = ROOT / "assets" / "test_data"
        actual_counts = {
            directory.name: len(list(directory.glob("*.png")))
            for directory in test_data_dir.iterdir()
            if directory.is_dir()
        }
        self.assertEqual(actual_counts, expected_counts)
        self.assertEqual(sum(actual_counts.values()), 519)

    def test_builtin_test_data_renders_without_errors(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.segmented_control[0].value, "Built-in test data")
        self.assertEqual(len(app.file_uploader), 0)
        self.assertEqual(len(app.selectbox), 2)
        self.assertEqual(app.selectbox[0].value, "Cracked")
        rendered = "\n".join(block.value for block in app.markdown)
        self.assertIn("Predicted class", rendered)
        self.assertIn("Class scores", rendered)

    def test_upload_state_renders_without_errors(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.segmented_control[0].set_value("Upload your image").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.selectbox), 0)


if __name__ == "__main__":
    unittest.main()
