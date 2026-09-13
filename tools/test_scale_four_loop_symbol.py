import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scale_four_loop_symbol import scale_row, verify_export


def row(numerator="1", denominator="32"):
    return {"word": ["a"] * 7 + ["mu"], "numerator": numerator, "denominator": denominator}


class ScalingTests(unittest.TestCase):
    def test_exact_scaling_and_source_unchanged(self):
        for n, d, expected in [("1", "32", "1"), ("-9", "16", "-18"),
                               ("6", "1", "192"), ("-15", "2", "-240"), ("60", "1", "1920")]:
            original = row(n, d)
            self.assertEqual(scale_row(original), row(expected, "1"))
            self.assertEqual(original, row(n, d))

    def test_reject_invalid_coefficients(self):
        for n, d in [("1", "3"), ("2", "4"), ("0", "1"), ("1", "-2"),
                     ("01", "2"), (1, "32"), ("1", "0")]:
            with self.subTest(n=n, d=d), self.assertRaises(ValueError):
                scale_row(row(n, d))

    def test_reject_invalid_words(self):
        for word in [["a"] * 7, ["unknown"] * 8]:
            original = row()
            original["word"] = word
            with self.assertRaises(ValueError):
                scale_row(original)

    def test_readback_detects_corruption_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gz", Path(directory) / "target.gz"
            with gzip.open(source, "wt") as stream:
                stream.write(json.dumps(row()) + "\n")
            for records, valid in [([row("1", "1")], True), ([row("2", "1")], False), ([], False),
                                   ([row("1", "1"), row("1", "1")], False)]:
                with gzip.open(target, "wt") as stream:
                    for record in records:
                        stream.write(json.dumps(record) + "\n")
                if valid:
                    self.assertEqual(verify_export(source, target), 1)
                else:
                    with self.assertRaises(ValueError):
                        verify_export(source, target)


if __name__ == "__main__":
    unittest.main()
