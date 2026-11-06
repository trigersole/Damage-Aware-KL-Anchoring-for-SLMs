import unittest

from kl_anchors.evaluate_math10k import load_configured_dataset


class Math10KEvaluationLoadingTests(unittest.TestCase):
    def test_pinned_parquet_loader_avoids_dataset_script(self):
        revision = "a" * 40
        config = {
            "repo_id": "example/dataset", "revision": revision,
            "split": "validation", "format": "parquet",
            "data_file": f"https://example.test/{revision}/validation.parquet",
        }
        calls = []

        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return "dataset"

        self.assertEqual(load_configured_dataset(config, loader=loader), "dataset")
        self.assertEqual(calls, [(('parquet',), {
            "data_files": {"validation": config["data_file"]},
            "split": "validation",
        })])

    def test_parquet_url_must_pin_configured_revision(self):
        with self.assertRaisesRegex(ValueError, "configured revision"):
            load_configured_dataset({
                "repo_id": "example/dataset", "revision": "a" * 40,
                "split": "validation", "format": "parquet",
                "data_file": "https://example.test/main/validation.parquet",
            }, loader=lambda *args, **kwargs: None)


if __name__ == "__main__":
    unittest.main()
