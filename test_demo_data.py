import csv
from pathlib import Path
import tempfile
import unittest
from make_demo_data import generate


class DemoDataTests(unittest.TestCase):
    def test_fixture_is_synthetic_and_does_not_overwrite_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "fixture.csv"
            generate(target, rows=30)
            before = target.read_bytes()
            with target.open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 30)
            self.assertIn("healthRiskScore", records[0])
            self.assertGreater(len(records[0]), 20)
            with self.assertRaises(FileExistsError):
                generate(target)
            self.assertEqual(before, target.read_bytes())


if __name__ == "__main__":
    unittest.main()
