import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from frame_metrics import FrameMetricsError, FrameSample, main, read_frames, summarize, summarize_csv


class FrameMetricsTests(unittest.TestCase):
    def test_asymmetric_tail_and_labels(self):
        csv = "frame_index,frame_ms,phase,event\n0,1.0,terrain,upload\n1,2.0,,\n2,10.0,ui,stall\n3,3.0,,\n"
        samples = read_frames(io.StringIO(csv))
        result = summarize(samples)
        self.assertEqual(result["sample_count"], 4)
        self.assertAlmostEqual(result["duration_ms"], 16.0)
        self.assertAlmostEqual(result["average_fps"], 250.0)
        self.assertAlmostEqual(result["mean_frame_ms"], 4.0)
        self.assertAlmostEqual(result["p50_frame_ms"], 2.5)
        self.assertAlmostEqual(result["p95_frame_ms"], 8.95)
        self.assertAlmostEqual(result["p99_frame_ms"], 9.79)
        self.assertAlmostEqual(result["p99_9_frame_ms"], 9.979)
        self.assertAlmostEqual(result["max_frame_ms"], 10.0)
        self.assertAlmostEqual(result["stddev_frame_ms"], math.sqrt(12.5))
        self.assertAlmostEqual(result["one_percent_low_fps"], 100.0)
        self.assertEqual(result["samples"][0]["phase"], "terrain")
        self.assertEqual(result["samples"][2]["event"], "stall")
        self.assertEqual(result["samples"][1]["phase"], "")

    def test_rejects_empty_and_missing_columns(self):
        for csv in ("", "frame_index,frame_ms\n", "frame_ms\n1\n", "frame_index\n0\n"):
            with self.assertRaises(FrameMetricsError):
                read_frames(io.StringIO(csv))

    def test_rejects_dropped_or_malformed_samples(self):
        cases = (
            "frame_index,frame_ms\n0,1\n2,1\n",
            "frame_index,frame_ms\n0,1\n1,nan\n",
            "frame_index,frame_ms\n0,inf\n",
            "frame_index,frame_ms\n0,0\n",
            "frame_index,frame_ms\n0,-1\n",
            "frame_index,frame_ms\n0,1\n1,oops\n",
            "frame_index,frame_ms\n0,1,extra\n",
            "frame_index,frame_ms,phase\n0,1,ok\n1\n",
        )
        for csv in cases:
            with self.subTest(csv=csv), self.assertRaises(FrameMetricsError):
                read_frames(io.StringIO(csv))

    def test_json_metrics_have_no_non_finite_values(self):
        result = summarize(read_frames(io.StringIO("frame_index,frame_ms\n0,1\n1,2\n")))
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_rejects_non_finite_aggregate(self):
        samples = read_frames(io.StringIO("frame_index,frame_ms\n0,1e308\n1,1e308\n"))
        with self.assertRaises(FrameMetricsError):
            summarize(samples)

    def test_direct_summary_revalidates_samples(self):
        bad = (
            [FrameSample(True, 1.0)],
            [FrameSample(0, float("nan"))],
            [FrameSample(0, 0.0)],
            [FrameSample(0, 1.0), FrameSample(2, 1.0)],
        )
        for samples in bad:
            with self.subTest(samples=samples), self.assertRaises(FrameMetricsError):
                summarize(samples)

    def test_rejects_strict_csv_and_truncated_optional_cells(self):
        for csv in (
            'frame_index,frame_ms\n0,"1\n',
            "frame_index,frame_ms,phase\n0,1\n",
        ):
            with self.subTest(csv=csv), self.assertRaises(FrameMetricsError):
                read_frames(io.StringIO(csv))

    def test_invalid_utf8_and_existing_output_are_user_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "frames.csv"
            input_path.write_bytes(b"frame_index,frame_ms\n0,1\n\xff")
            with self.assertRaises(FrameMetricsError):
                summarize_csv(input_path)

            input_path.write_text("frame_index,frame_ms\n0,1\n", encoding="utf-8")
            output_path = root / "summary.json"
            output_path.write_text("keep", encoding="utf-8")
            with self.assertRaises(SystemExit):
                main([str(input_path), "-o", str(output_path)])
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
