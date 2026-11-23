"""Timeline windows, sidecar equality and the empty-gate fix in quick_ab.py, on synthetic traces.

Run with the rest of the folder's tests:

    python -m unittest discover -s mc-rs/optimization -p test_*.py -v
"""

import contextlib
import csv
import io
import json
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

import quick_ab

WORK_COLUMNS = (
    "terrain_loaded_sections",
    "terrain_quads",
    "entity_vertices",
    "mesh_pending",
    "mesh_deferred",
    "mesh_in_flight",
    "mesh_ready_uploads",
)

WINDOW_PLAN = {
    "batch": "synthetic-window",
    "window_phase": "sw_fight",
    "window_frames_expected": 1000,
    "tail_phase": "sw_tail",
    "window_sum_counters": ["entity_vertices"],
    "stable_counters": ["terrain_loaded_sections", "terrain_quads"],
}


def frame(ms, phase, loaded=100, busy=0, quads=50, vertices=10, event="ok"):
    return {
        "ms": ms,
        "phase": phase,
        "event": event,
        "terrain_loaded_sections": loaded,
        "terrain_quads": quads,
        "entity_vertices": vertices,
        "mesh_pending": busy,
    }


def scene(loading=50, join=1500, join_busy=300, fight=1000, tail=200, trailing=5):
    """Loading, then a join whose first `join_busy` frames mesh, the window, a tail, and frames
    after the tail whose counters differ, so a test can tell where final_work was read."""
    rows = [frame(5.0, "loading", loaded=0) for _ in range(loading)]
    rows += [frame(2.0, "sw_join", busy=3 if i < join_busy else 0) for i in range(join)]
    rows += [frame(1.0 + (i % 10) * 0.1, "sw_fight", busy=1 if i % 4 == 0 else 0) for i in range(fight)]
    rows += [frame(1.5, "sw_tail") for _ in range(tail)]
    if rows[-1]["phase"] == "sw_tail":
        rows[-1]["terrain_quads"] = 777
    rows += [frame(1.5, "after", quads=999) for _ in range(trailing)]
    return rows


def write_run(run_dir, rows, sidecar=None, meta=None, record=None):
    run_dir.mkdir(parents=True)
    with (run_dir / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "frame_ms", "phase", "event"])
        for index, row in enumerate(rows):
            writer.writerow([index, row["ms"], row["phase"], row["event"]])
    with (run_dir / "frames.csv.phases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "ticks_ms", "render_ms"])
        for index, row in enumerate(rows):
            writer.writerow([index, 0.25, row["ms"] / 2])
    with (run_dir / "frames.csv.work.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", *WORK_COLUMNS])
        for index, row in enumerate(rows):
            writer.writerow([index, *(row.get(column, 0) for column in WORK_COLUMNS)])
    if sidecar is not None:
        with (run_dir / quick_ab.SIDECAR_NAME).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "step", "tick", "segment", "particles"])
            writer.writerows(sidecar)
    if meta is not None:
        (run_dir / "frames.csv.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if record is not None:
        (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")


def sidecar_rows(offset=0, particles_override=None):
    """Lockstep rows; `frame` depends on the loading length and must be ignored."""
    rows = []
    for step in range(1, 101):
        particles = 3 * step
        if particles_override and step in particles_override:
            particles = particles_override[step]
        rows.append([step + 50 + offset, step, step // 50, "sw_fight", particles])
    return rows


def analyze_quietly(output):
    with contextlib.redirect_stdout(io.StringIO()):
        return quick_ab.analyze(output)


class PhaseBlockTests(unittest.TestCase):
    def test_contiguous_missing_and_split(self):
        labels = ["loading", "a", "a", "b", "b", "b", "c"]
        self.assertEqual(quick_ab.phase_block(labels, "b"), (3, 6))
        self.assertIn("not found", quick_ab.phase_block(labels, "z"))
        split = quick_ab.phase_block(["a", "b", "a"], "a")
        self.assertIn("not one contiguous block", split)


class WindowSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def summarize(self, rows, plan=None, **kwargs):
        run_dir = self.root / f"run{len(list(self.root.iterdir()))}"
        write_run(run_dir, rows, **kwargs)
        return quick_ab.summarize(run_dir, plan or WINDOW_PLAN)

    def test_window_load_settle_slices_sums_and_tail(self):
        rows = scene()
        summary = self.summarize(rows)
        self.assertNotIn("window_invalid", summary)
        self.assertEqual((summary["window_start"], summary["window_end"]), (1550, 2550))
        self.assertEqual(summary["window_frames"], 1000)
        fight_ms = [1.0 + (i % 10) * 0.1 for i in range(1000)]
        self.assertAlmostEqual(summary["window"]["mean"], statistics.fmean(fight_ms))
        self.assertAlmostEqual(summary["window"]["max"], 1.9)
        self.assertEqual(summary["window"]["n"], 1000)
        # The settle search sees only pre-window frames: loaded from frame 50, busy for 300.
        self.assertEqual(summary["first_loaded_frame"], 50)
        self.assertEqual(summary["settled_frame"], 350)
        self.assertAlmostEqual(summary["load_ms"], 600.0)
        self.assertEqual(len(summary["window_slices"]), 10)
        self.assertEqual(summary["window_slices"][0]["start"], 1550)
        self.assertEqual(summary["window_slices"][-1]["end"], 2550)
        self.assertTrue(all(block["n"] == 100 for block in summary["window_slices"]))
        self.assertAlmostEqual(summary["window_busy_fraction"], 0.25)
        self.assertAlmostEqual(summary["window_phases"]["ticks_ms"], 0.25)
        self.assertEqual(summary["window_sums"], {"entity_vertices": 10_000})
        # final_work comes from the tail's last frame, not from the frames after it.
        self.assertEqual(summary["final_work"], {"terrain_loaded_sections": 100, "terrain_quads": 777})
        self.assertEqual((summary["tail_start"], summary["tail_end"]), (2550, 2750))
        self.assertNotIn("steady", summary)

    def test_without_tail_phase_final_work_is_the_last_frame(self):
        plan = {**WINDOW_PLAN, "tail_phase": None}
        summary = self.summarize(scene(), plan)
        self.assertEqual(summary["final_work"]["terrain_quads"], 999)

    def test_invalid_windows(self):
        cases = {
            "not found": [frame(1.0, "sw_join") for _ in range(10)],
            "not one contiguous block": scene()[:1600] + [frame(1.0, "sw_join")] + scene()[1600:],
            "expected 1000": scene(fight=999),
            "did not settle": scene(join_busy=1400),
            "acquire skips": [
                {**row, "event": "acquire_skips=1"} if index == 2000 else row
                for index, row in enumerate(scene())
            ],
            "'sw_tail' not found": scene(tail=0),
        }
        for needle, rows in cases.items():
            with self.subTest(needle):
                summary = self.summarize(rows)
                self.assertIn(needle, summary.get("window_invalid", ""))
        overflow = self.summarize(scene(), meta={"overflowed": True})
        self.assertIn("overflowed", overflow["window_invalid"])

    def test_missing_counter_cells_are_not_zero(self):
        rows = scene()
        rows[2000]["entity_vertices"] = ""
        summary = self.summarize(rows)
        self.assertIsNone(summary["window_sums"]["entity_vertices"])

    def test_old_plans_take_the_old_path(self):
        plan = {"batch": "old", "steady_frames": 500, "stable_counters": ["terrain_quads"]}
        summary = self.summarize(scene(), plan)
        for key in ("window", "window_start", "window_invalid", "window_slices", "sidecar"):
            self.assertNotIn(key, summary)
        # Old rule: settle over the whole trace, steady = the last 500 frames, final_work from
        # the last row with empty cells read as zero.
        self.assertEqual(summary["settled_frame"], 350)
        self.assertEqual(summary["steady_start"], 2755 - 500)
        self.assertTrue(summary["steady_after_settle"])
        self.assertEqual(summary["final_work"], {"terrain_quads": 999})


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "batch"
        self.output.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, plan, rows_for, sidecar_for=None):
        (self.output / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (self.output / "identity.json").write_text(json.dumps({"A": "x", "B": "x"}), encoding="utf-8")
        for pair in (0, 1):
            for lane in "AB":
                run = f"p{pair:02d}-{lane}"
                write_run(
                    self.output / run,
                    rows_for(run),
                    sidecar=sidecar_for(run) if sidecar_for else None,
                    record={"run": run, "pair": pair, "lane": lane, "returncode": 0, "wall_s": 1.0},
                )
        report = analyze_quietly(self.output)
        markdown = (self.output / "analysis.md").read_text(encoding="utf-8")
        return report, markdown

    def test_identical_window_runs_pass_every_gate(self):
        # Each run's `frame` column differs (different loading lengths); it must be ignored.
        report, markdown = self.batch(
            WINDOW_PLAN, lambda run: scene(), lambda run: sidecar_rows(offset=ord(run[-1]) + int(run[2]))
        )
        self.assertEqual(report["pairs_complete"], [0, 1])
        self.assertTrue(report["settled_work_identical"])
        self.assertTrue(report["same_work_identical"])
        self.assertEqual(set(report["same_work"]), {"final_work", "window_sums", "sidecar"})
        self.assertEqual(report["same_work"]["sidecar"]["rows"], [100])
        self.assertIn("window.mean", report["comparisons"])
        self.assertIn("window_phase.ticks_ms", report["comparisons"])
        self.assertNotIn("steady.mean", report["comparisons"])
        self.assertEqual(len(report["window_slice_comparisons"]), 10)
        self.assertNotIn("settled_work_unchecked", report)
        self.assertIn("Same work: PASS", markdown)

    def test_sidecar_divergence_is_located(self):
        def sidecar_for(run):
            return sidecar_rows(particles_override={7: 0} if run == "p01-B" else None)

        report, markdown = self.batch(WINDOW_PLAN, lambda run: scene(), sidecar_for)
        gate = report["same_work"]["sidecar"]
        self.assertFalse(gate["identical"])
        self.assertEqual(gate["distinct_digests"], 2)
        divergence = gate["first_divergence"]
        self.assertEqual((divergence["run"], divergence["row"]), ("p01-B", 6))
        self.assertEqual((divergence["column"], divergence["reference"], divergence["value"]), ("particles", "21", "0"))
        self.assertFalse(report["same_work_identical"])
        self.assertIn("Same work: FAIL", markdown)

    def test_window_sum_mismatch_and_invalid_run(self):
        def rows_for(run):
            rows = scene()
            if run == "p00-B":
                rows[2000]["entity_vertices"] = 11
            if run == "p01-A":
                rows = scene(fight=900)  # wrong window length: invalid, so pair 1 is incomplete
            return rows

        report, _ = self.batch(WINDOW_PLAN, rows_for)
        self.assertEqual(report["invalid_runs"], ["p01-A"])
        self.assertEqual(report["pairs_complete"], [0])
        sums = report["same_work"]["window_sums"]
        self.assertFalse(sums["identical"])
        self.assertEqual(sums["values"]["entity_vertices"], [10_000, 10_001])
        self.assertNotIn("sidecar", report["same_work"])

    def test_old_plan_that_never_settles_fails_the_gate_explicitly(self):
        plan = {"batch": "old-busy", "steady_frames": 100, "stable_counters": ["terrain_quads"]}
        busy_forever = [frame(1.0, "replay", busy=1) for _ in range(300)]
        report, markdown = self.batch(plan, lambda run: busy_forever)
        self.assertEqual(report["pairs_complete"], [0, 1])
        self.assertEqual(report["settled_work_values"], {})
        self.assertFalse(report["settled_work_identical"])
        self.assertEqual(
            report["settled_work_unchecked"]["runs_without_final_work"],
            ["p00-A", "p00-B", "p01-A", "p01-B"],
        )
        self.assertIn("Same-work gate FAILED: nothing was compared", markdown)
        self.assertNotIn("window", report)

    def test_old_plan_that_settles_keeps_the_old_report_shape(self):
        plan = {"batch": "old-quiet", "steady_frames": 500, "stable_counters": ["terrain_quads"]}
        report, markdown = self.batch(plan, lambda run: scene())
        self.assertTrue(report["settled_work_identical"])
        self.assertEqual(
            set(report),
            {
                "batch",
                "lane_env",
                "identical_executables",
                "pairs_complete",
                "invalid_runs",
                "settled_work_values",
                "settled_work_identical",
                "comparisons",
                "per_run",
            },
        )
        self.assertIn("steady.mean", report["comparisons"])
        self.assertNotIn("Same-work gate FAILED", markdown)


# A stand-in client. Like the real one it writes its trace where
# MC_FRAME_TRACE says, resolved against its own working directory (the run directory), and it
# fails when that folder does not exist; it also refuses a relative --gameDir.
FAKE_CLIENT = """
import csv, os, sys
game_dir = sys.argv[sys.argv.index("--gameDir") + 1]
if not os.path.isabs(game_dir):
    sys.exit(9)
trace = os.environ["MC_FRAME_TRACE"]
with open(trace, "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame_index", "frame_ms", "phase", "event"])
    for index in range(1200):
        writer.writerow([index, 1.0, "run", "ok"])
with open(trace + ".work.csv", "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame_index", "terrain_loaded_sections", "mesh_pending"])
    for index in range(1200):
        writer.writerow([index, 1, 0])
"""


class FakeLease:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def heartbeat(self):
        pass


class FakeWatch:
    def __init__(self, pid, send_to_back):
        self.moved_back = False
        self.samples = 0
        self.foreground_samples = 0

    def poll(self):
        self.samples += 1


class RelativeOutputTests(unittest.TestCase):
    """A relative --output used to give a batch of empty runs."""

    def run_in(self, root, plan_path, output):
        saved = (quick_ab.BenchmarkLease, quick_ab.WindowWatch, quick_ab.host_notes, os.getcwd())
        quick_ab.BenchmarkLease = FakeLease
        quick_ab.WindowWatch = FakeWatch
        quick_ab.host_notes = lambda: {}
        os.chdir(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                quick_ab.run_batch(plan_path, output)
        finally:
            quick_ab.BenchmarkLease, quick_ab.WindowWatch, quick_ab.host_notes, cwd = saved
            os.chdir(cwd)

    def test_a_relative_output_still_records_every_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = root / "fake_client.py"
            client.write_text(FAKE_CLIENT, encoding="utf-8")
            plan = {
                "batch": "relative-output",
                "lanes": {"A": sys.executable, "B": sys.executable},
                "pairs": 1,
                "warmup": False,
                "timeout_s": 120,
                "args": [str(client)],
                "steady_frames": 100,
            }
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            self.run_in(root, Path("plan.json"), Path("out"))
            for run in ("p00-A", "p00-B"):
                record = json.loads((root / "out" / run / "record.json").read_text(encoding="utf-8"))
                self.assertEqual(record["returncode"], 0, run)
                self.assertEqual(record["summary"]["frames"], 1200, run)
                self.assertTrue(Path(record["command"][-1]).is_absolute(), run)
            report = json.loads((root / "out" / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(report["pairs_complete"], [0])
            self.assertEqual(report["invalid_runs"], [])

    def test_relative_paths_in_args_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.mccap").write_bytes(b"")
            cwd = os.getcwd()
            os.chdir(root)
            try:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    found = quick_ab.warn_relative_args(
                        ["--replay", "scene.mccap", "--frames", "60000", str(root / "scene.mccap")]
                    )
            finally:
                os.chdir(cwd)
            self.assertEqual(found, ["scene.mccap"])
            self.assertIn("relative path", err.getvalue())


if __name__ == "__main__":
    unittest.main()
