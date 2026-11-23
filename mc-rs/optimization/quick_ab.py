"""Paired control/candidate timing runs of the real client through `--replay`.

This is the lighter process from the October 18 direction reset: one record per
measured run (executable SHA-256, config, options and host notes), balanced ABBA
order, an A/A noise baseline when both lanes name the same executable, and
run-level 95% intervals over every valid run. It reports ms/frame, never FPS.

Run it with Windows Python, because the client and the benchmark lease are native:

    python quick_ab.py run --plan plan.json --output <dir>
    python quick_ab.py analyze --output <dir>

`--output` may be relative (it is resolved first); paths inside the plan's `args` must be
absolute, because the client runs with its run directory as the working directory (quick_ab
warns about relative ones that exist here).

A plan is JSON:

    {
      "batch": "pf1-signed-e1-100",
      "lanes": {"A": "path/to/control.exe", "B": "path/to/candidate.exe"},
      "pairs": 4,
      "timeout_s": 900,
      "args": ["--replay", "scene.mccap", "--replayCamera", "fixed", "--frames", "60000"],
      "options": {"renderDistance": "8"},
      "env": {},
      "lane_env": {"B": {"NAME": "value"}},
      "steady_frames": 20000,
      "stable_counters": ["terrain_quads", "terrain_visible_sections", "entity_vertices"]
    }

The ABBA schedule is fixed up front, so a slow run cannot reorder anything, and a
failed run stays in the record instead of being replaced.

Timeline windows (optional; a plan without `window_phase` behaves exactly as before). For a
lockstep scene such as `skywars_bench`, where frame i is the same scene in every build, the
measured window comes from the scene's own timeline instead of from the last N frames:

    "window_phase": "sw_fight",        the window is the one contiguous block of frames whose
                                       frames.csv `phase` is this label; missing, split, or
                                       containing acquire skips makes the run invalid
    "window_frames_expected": 60000,   optional: any other window length makes the run invalid
    "tail_phase": "sw_tail",           optional: read final_work on the last frame of this block
                                       (default: the last frame of the trace)
    "window_sum_counters": ["entity_vertices"],  optional: per-run sums over the window that
                                       must be identical across runs
    "window_slices": 10,               optional: the window is also reported in this many slices
    "sidecar_ignore": ["frame"]        optional: sidecar columns left out of the equality check

The load/settle search then only covers frames before the window, and a run whose mesh queues
do not settle before the window is invalid. If a run directory has `frames.csv.sidecar.csv`
(written by `skywars_bench`), every run's sidecar rows must be identical.

The same-work gate fails when it has nothing to compare: a run without `final_work` (it never
settled, so it had no steady window) or a batch without complete pairs. It used to pass then.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_lease import BenchmarkLease  # noqa: E402

# Two-sided 97.5% Student t quantiles for small run counts (df -> t).
T_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    15: 2.1314,
    19: 2.0930,
}

# Benchmark settings. Everything else keeps the client's defaults. These mirror the
# user's own options.txt where they affect rendering cost; the user's file is never read
# at run time and is never written. Focus loss must not open the pause menu mid-run.
BASE_OPTIONS = {
    "renderDistance": "8",
    "guiScale": "3",
    "particles": "1",
    "bobView": "false",
    "maxFps": "260",
    "enableVsync": "false",
    "fancyGraphics": "true",
    "ao": "2",
    "renderClouds": "fast",
    "mipmapLevels": "4",
    "entityShadows": "true",
    "pauseOnLostFocus": "false",
    "fov": "0.5",
    "msaaSamples": "1",
    "presentMode": "auto",
    "soundCategory_master": "0.0",
    "soundCategory_music": "0.0",
}


def checkout_jars_dir() -> str | None:
    """This checkout's vanilla 1.8.9 files, the default MC_JARS_DIR for every run.

    The plans pass no --versionJar, and an optimised mc-client has no path to the checkout
    compiled in: it would name a folder of the PC that built it. Builds from before that read
    MC_JARS_DIR first too, so both arms of an A/B load the
    same files. An MC_JARS_DIR set by the caller or the plan wins.
    """
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "MCP-919-main" / "jars", root / "jars"):
        if candidate.is_dir():
            return str(candidate)
    return None


QUEUE_COUNTERS = ("mesh_pending", "mesh_deferred", "mesh_in_flight", "mesh_ready_uploads")
SETTLE_QUIET_FRAMES = 1000
DEFAULT_STABLE_COUNTERS = (
    "terrain_loaded_sections",
    "terrain_visible_sections",
    "terrain_draw_calls",
    "terrain_quads",
    "terrain_resident_bytes",
    "entity_vertices",
    "entity_batches",
)
SIDECAR_NAME = "frames.csv.sidecar.csv"
METRICS = ("mean", "p50", "p95", "p99", "p999", "max")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation on the inclusive rank, matching frame_metrics.py."""
    if not sorted_values:
        return math.nan
    rank = (len(sorted_values) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered) if ordered else math.nan,
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "p999": percentile(ordered, 0.999),
        "max": ordered[-1] if ordered else math.nan,
    }


def host_notes() -> dict[str, object]:
    """Record competing work. Nothing here stops or changes a process."""
    notes: dict[str, object] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        output = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
        names = [row[0].lower() for row in csv.reader(output.splitlines()) if row]
        watched = ("java", "javaw", "cargo", "rustc", "mc-client", "blender", "optimization_")
        notes["watched_processes"] = sorted(
            {name: names.count(name) for name in names if name.startswith(watched)}.items()
        )
    except (OSError, subprocess.SubprocessError) as error:
        notes["watched_processes_error"] = str(error)
    return notes


class WindowWatch:
    """Keep a test window out of the way, and record whether it ever holds focus.

    The client itself handles the part that matters: `MC_BACKGROUND_WINDOW` stops it activating
    its window, capturing the cursor or reading input. This adds two conveniences from outside: a
    click cannot activate the window, and it sits behind whatever the person is working on.
    """

    GWL_EXSTYLE = -20
    WS_EX_NOACTIVATE = 0x08000000
    HWND_BOTTOM = 1
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOACTIVATE = 0x0010

    def __init__(self, pid: int, send_to_back: bool) -> None:
        self.pid = pid
        self.send_to_back = send_to_back
        self.guarded: set[int] = set()
        self.moved_back = False
        self.samples = 0
        self.foreground_samples = 0
        self.next_enumeration = 0.0
        self.user32 = ctypes.windll.user32
        self.user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        self.user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        self.user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p

    def _window_pid(self, hwnd) -> int:
        pid = ctypes.c_ulong(0)
        self.user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
        return pid.value

    def _own_windows(self) -> list[int]:
        found: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def visit(hwnd, _param):
            if self._window_pid(hwnd) == self.pid:
                found.append(hwnd)
            return True

        self.user32.EnumWindows(visit, None)
        return found

    def _guard_window(self, hwnd: int) -> None:
        """Runs on its own thread: these calls send messages to the game's window thread, which
        pumps nothing during startup asset loading and can block them for many seconds."""
        started = time.monotonic()
        handle = ctypes.c_void_p(hwnd)
        style = self.user32.GetWindowLongPtrW(handle, self.GWL_EXSTYLE)
        self.user32.SetWindowLongPtrW(handle, self.GWL_EXSTYLE, style | self.WS_EX_NOACTIVATE)
        if not self.send_to_back:
            return
        while not self.user32.IsWindowVisible(handle):
            if not self.user32.IsWindow(handle) or time.monotonic() - started > 60:
                return
            time.sleep(0.01)
        self.user32.SetWindowPos(
            handle,
            ctypes.c_void_p(self.HWND_BOTTOM),
            0,
            0,
            0,
            0,
            self.SWP_NOSIZE | self.SWP_NOMOVE | self.SWP_NOACTIVATE,
        )
        self.moved_back = True

    def poll(self) -> None:
        self.samples += 1
        now = time.monotonic()
        if now >= self.next_enumeration:
            self.next_enumeration = now + 0.05
            for hwnd in self._own_windows():
                if hwnd not in self.guarded:
                    self.guarded.add(hwnd)
                    threading.Thread(target=self._guard_window, args=(hwnd,), daemon=True).start()
        foreground = self.user32.GetForegroundWindow()
        if foreground and self._window_pid(foreground) == self.pid:
            self.foreground_samples += 1


def write_options(game_dir: Path, overrides: dict[str, str]) -> Path:
    game_dir.mkdir(parents=True, exist_ok=True)
    merged = {**BASE_OPTIONS, **overrides}
    path = game_dir / "options.txt"
    path.write_text("".join(f"{key}:{value}\n" for key, value in merged.items()), encoding="utf-8")
    return path


def abba_schedule(pairs: int) -> list[tuple[int, str]]:
    schedule: list[tuple[int, str]] = []
    for pair in range(pairs):
        order = "AB" if pair % 2 == 0 else "BA"
        schedule.extend((pair, lane) for lane in order)
    return schedule


def read_trace(run_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    def load(name: str) -> list[dict]:
        path = run_dir / name
        if not path.exists():
            gz_path = run_dir / (name + ".gz")
            if not gz_path.exists():
                return []
            with gzip.open(gz_path, "rt", encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    return load("frames.csv"), load("frames.csv.phases.csv"), load("frames.csv.work.csv")


def as_int(row: dict, key: str) -> int:
    raw = row.get(key)
    return int(raw) if raw not in (None, "") else 0


def optional_int(row: dict, key: str) -> int | None:
    """Like `as_int`, but an empty cell (instrumentation unavailable) stays missing."""
    raw = row.get(key)
    return int(raw) if raw not in (None, "") else None


def phase_block(labels: list[str], name: str) -> tuple[int, int] | str:
    """The one contiguous run of frames labelled `name`, as [start, end), or why there is none."""
    hits = [index for index, label in enumerate(labels) if label == name]
    if not hits:
        return f"phase {name!r} not found"
    start, end = hits[0], hits[-1] + 1
    if end - start != len(hits):
        return (
            f"phase {name!r} is not one contiguous block "
            f"({len(hits)} frames spread over [{start}, {end}))"
        )
    return start, end


def read_meta(run_dir: Path) -> dict:
    path = run_dir / "frames.csv.meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_sidecar(run_dir: Path) -> list[dict] | None:
    """`skywars_bench`'s per-frame sidecar, plain or gzipped; None when the run has none."""
    path = run_dir / SIDECAR_NAME
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    gz_path = run_dir / (SIDECAR_NAME + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    return None


def sidecar_columns(rows: list[dict], ignore: list[str]) -> list[str]:
    return [name for name in (rows[0] if rows else {}) if name not in ignore]


def sidecar_digest(rows: list[dict], ignore: list[str]) -> dict:
    columns = sidecar_columns(rows, ignore)
    digest = hashlib.sha256()
    for row in rows:
        digest.update((",".join(row.get(name) or "" for name in columns) + "\n").encode("utf-8"))
    return {"rows": len(rows), "columns": columns, "sha256": digest.hexdigest()}


def settle_search(work: list[dict]) -> tuple[int | None, int | None, int]:
    """First loaded frame, settle frame (None if never) and busy frames after it, over `work`."""
    first_loaded = None
    settled_at = None
    quiet = 0
    later_busy = 0
    for index, row in enumerate(work):
        if first_loaded is None and as_int(row, "terrain_loaded_sections") > 0:
            first_loaded = index
        if first_loaded is None:
            continue
        busy = sum(as_int(row, key) for key in QUEUE_COUNTERS) > 0
        if settled_at is None:
            quiet = 0 if busy else quiet + 1
            if quiet >= SETTLE_QUIET_FRAMES:
                settled_at = index + 1 - SETTLE_QUIET_FRAMES
        elif busy:
            later_busy += 1
    return first_loaded, settled_at, later_busy


def summarize_window(run_dir: Path, plan: dict) -> dict:
    """One run's summary for a timeline-window plan (`window_phase`; see the module docstring).

    The run is invalid, with the reasons in `window_invalid`, when the window block is missing,
    split, the wrong length, or follows an acquire skip; when the trace overflowed; when the mesh
    queues did not settle before the window; or when a named `tail_phase` block is missing.
    """
    frames, phases, work = read_trace(run_dir)
    summary: dict[str, object] = {"frames": len(frames)}
    if not frames:
        summary["window_invalid"] = "no frames"
        return summary
    frame_ms = [float(row["frame_ms"]) for row in frames]
    summary["total_ms"] = sum(frame_ms)
    if len(work) != len(frames):
        summary["work_rows"] = len(work)
    labels = [row.get("phase") or "" for row in frames]
    block = phase_block(labels, plan["window_phase"])
    if isinstance(block, str):
        summary["window_invalid"] = block
        return summary
    start, end = block
    problems: list[str] = []
    summary["window_start"] = start
    summary["window_end"] = end
    summary["window_frames"] = end - start
    expected = plan.get("window_frames_expected")
    if expected is not None and end - start != int(expected):
        problems.append(f"window has {end - start} frames, expected {int(expected)}")
    # A skipped acquire still advanced the lockstep scene, so later frames no longer pair up.
    skipped = [i for i in range(start, end) if (frames[i].get("event") or "ok") != "ok"]
    if skipped:
        problems.append(f"{len(skipped)} window frames follow acquire skips (first: frame {skipped[0]})")
    if read_meta(run_dir).get("overflowed"):
        problems.append("the frame trace overflowed")

    first_loaded, settled_at, later_busy = settle_search(work[:start])
    summary["first_loaded_frame"] = first_loaded
    summary["settled_frame"] = settled_at
    summary["busy_frames_after_settle"] = later_busy
    if settled_at is None:
        summary["never_settled"] = True
        problems.append("the mesh queues did not settle before the window")
    elif first_loaded is not None and settled_at > first_loaded:
        load = frame_ms[first_loaded:settled_at]
        summary["load_ms"] = sum(load)
        summary["load"] = distribution(load)

    summary["window"] = distribution(frame_ms[start:end])
    if len(phases) >= end:
        names = [key for key in phases[0] if key.endswith("_ms")]
        summary["window_phases"] = {
            name: statistics.fmean(float(row[name] or 0.0) for row in phases[start:end])
            for name in names
        }
    window_work = work[start:end] if len(work) >= end else []
    if window_work:
        busy = sum(1 for row in window_work if sum(as_int(row, key) for key in QUEUE_COUNTERS) > 0)
        summary["window_busy_frames"] = busy
        summary["window_busy_fraction"] = busy / len(window_work)
    count = max(1, min(int(plan.get("window_slices", 10)), end - start))
    bounds = [start + (end - start) * i // count for i in range(count + 1)]
    summary["window_slices"] = [
        {"start": low, "end": high, **distribution(frame_ms[low:high])}
        for low, high in zip(bounds, bounds[1:])
    ]
    if plan.get("window_sum_counters"):
        sums: dict[str, int | None] = {}
        for key in plan["window_sum_counters"]:
            values = [optional_int(row, key) for row in window_work]
            sums[key] = None if not values or None in values else sum(values)
        summary["window_sums"] = sums

    final_row = None
    tail = plan.get("tail_phase")
    if tail:
        tail_block = phase_block(labels, tail)
        if isinstance(tail_block, str):
            problems.append(tail_block)
        else:
            summary["tail_start"], summary["tail_end"] = tail_block
            if len(work) >= tail_block[1]:
                final_row = work[tail_block[1] - 1]
    elif work:
        final_row = work[-1]
    if final_row is not None:
        summary["final_work"] = {
            key: optional_int(final_row, key)
            for key in plan.get("stable_counters", DEFAULT_STABLE_COUNTERS)
        }
    sidecar = read_sidecar(run_dir)
    if sidecar is not None:
        summary["sidecar"] = sidecar_digest(sidecar, plan.get("sidecar_ignore", ["frame"]))
    if problems:
        summary["window_invalid"] = "; ".join(problems)
    return summary


def summarize(run_dir: Path, plan: dict) -> dict:
    """Derive one run's load and steady windows from its raw trace.

    Settled means the first frame from which every mesh queue counter stays at zero for
    SETTLE_QUIET_FRAMES frames, so one later block update in a capture does not count as still
    loading. Any later busy frames are reported separately. The steady window is the last
    `steady_frames` frames, and it only counts if it starts after the settle point; otherwise the
    run reports no steady numbers rather than mixing loading frames into them.

    A plan with `window_phase` is summarized by `summarize_window` instead.
    """
    if plan.get("window_phase"):
        return summarize_window(run_dir, plan)
    frames, phases, work = read_trace(run_dir)
    summary: dict[str, object] = {"frames": len(frames)}
    if not frames:
        return summary
    frame_ms = [float(row["frame_ms"]) for row in frames]
    summary["total_ms"] = sum(frame_ms)
    if len(work) != len(frames):
        summary["work_rows"] = len(work)

    first_loaded = None
    settled_at = None
    quiet = 0
    later_busy = 0
    for index, row in enumerate(work):
        if first_loaded is None and as_int(row, "terrain_loaded_sections") > 0:
            first_loaded = index
        if first_loaded is None:
            continue
        busy = sum(as_int(row, key) for key in QUEUE_COUNTERS) > 0
        if settled_at is None:
            quiet = 0 if busy else quiet + 1
            if quiet >= SETTLE_QUIET_FRAMES:
                settled_at = index + 1 - SETTLE_QUIET_FRAMES
        elif busy:
            later_busy += 1
    if settled_at is None:
        settled_at = len(frame_ms)
        summary["never_settled"] = True
    summary["first_loaded_frame"] = first_loaded
    summary["settled_frame"] = settled_at
    summary["busy_frames_after_settle"] = later_busy
    if first_loaded is not None and settled_at > first_loaded:
        window = frame_ms[first_loaded:settled_at]
        summary["load_ms"] = sum(window)
        summary["load"] = distribution(window)

    steady_frames = int(plan.get("steady_frames", 0))
    start = max(0, len(frame_ms) - steady_frames) if steady_frames else settled_at
    summary["steady_start"] = start
    summary["steady_after_settle"] = start >= settled_at
    if start >= settled_at and start < len(frame_ms):
        summary["steady"] = distribution(frame_ms[start:])
        if phases:
            names = [key for key in phases[0] if key.endswith("_ms")]
            summary["steady_phases"] = {
                name: statistics.fmean(float(row[name] or 0.0) for row in phases[start:])
                for name in names
            }
        if work:
            last = work[-1]
            summary["final_work"] = {
                key: as_int(last, key)
                for key in plan.get(
                    "stable_counters",
                    [
                        "terrain_loaded_sections",
                        "terrain_visible_sections",
                        "terrain_draw_calls",
                        "terrain_quads",
                        "terrain_resident_bytes",
                        "entity_vertices",
                        "entity_batches",
                    ],
                )
            }
    return summary


def run_batch(plan_path: Path, output: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # The client runs with its run directory as the working
    # directory, so every path quick_ab hands it must be absolute. A relative --output used to
    # make MC_FRAME_TRACE and --gameDir point inside the run directory itself: the trace's
    # folder did not exist, the bench exited at start-up and every run of the batch was empty.
    output = Path(output).resolve()
    warn_relative_args(plan.get("args", []))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan_path, output / "plan.json")
    lanes = {name: Path(path).resolve() for name, path in plan["lanes"].items()}
    identity = {name: sha256_file(path) for name, path in lanes.items()}
    (output / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    schedule = abba_schedule(int(plan.get("pairs", 4)))
    records = []
    # A warmup run per batch settles disk caches and shader caches so the first scored run
    # is not systematically slower. It is recorded and never scored.
    warmup = [(-1, "A")] if plan.get("warmup", True) else []
    with BenchmarkLease(owner=str(plan["batch"]), timeout=int(plan.get("lease_timeout_s", 60))) as lease:
        for pair, lane in warmup + schedule:
            lease.heartbeat()
            run_id = "warmup" if pair < 0 else f"p{pair:02d}-{lane}"
            run_dir = output / run_id
            if run_dir.exists():
                raise SystemExit(f"refusing to overwrite existing run {run_dir}")
            game_dir = run_dir / "game"
            options = write_options(game_dir, plan.get("options", {}))
            # Optional: files copied into every run's fresh game
            # dir before launch, as {"name in game dir": "source path"}, e.g. a mod settings file
            # {"mc-rs-mods.json": "..."}. "lane_game_files": {"B": {...}} adds or overrides files
            # for one lane. A plan without either key behaves exactly as before.
            game_files = {
                **plan.get("game_files", {}),
                **plan.get("lane_game_files", {}).get(lane, {}),
            }
            for name, source in game_files.items():
                shutil.copyfile(source, game_dir / name)
            env = dict(os.environ)
            jars = checkout_jars_dir()
            if jars is not None:
                env.setdefault("MC_JARS_DIR", jars)
            # Benchmark runs are offline and deterministic. The client
            # starts no skin download worker (profile skins stay the defaults; nothing is fetched
            # or written to the assets' skin cache). A plan may set "0" to restore it.
            env.setdefault("MC_NO_SKIN_DOWNLOADS", "1")
            env.update(plan.get("env", {}))
            env.update(plan.get("lane_env", {}).get(lane, {}))
            env["MC_FRAME_TRACE"] = str(run_dir / "frames.csv")
            env["MC_BACKGROUND_WINDOW"] = "1"
            env["MC_FRAME_TRACE_MAX_SAMPLES"] = str(plan.get("max_samples", 1_000_000))
            command = [str(lanes[lane]), *plan["args"], "--gameDir", str(game_dir)]
            record = {
                "run": run_id,
                "pair": pair,
                "lane": lane,
                "executable_sha256": identity[lane],
                "command": command,
                "options_sha256": sha256_file(options),
                "game_files_sha256": {name: sha256_file(game_dir / name) for name in game_files},
                "env_overrides":{**plan.get("env", {}), **plan.get("lane_env", {}).get(lane, {})},
                "host_before": host_notes(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            started = time.monotonic()
            timeout = float(plan.get("timeout_s", 900))
            with (run_dir / "stdout.log").open("wb") as out, (run_dir / "stderr.log").open("wb") as err:
                process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=out, stderr=err)
                watch = WindowWatch(process.pid, bool(plan.get("send_to_back", True)))
                while process.poll() is None:
                    watch.poll()
                    if time.monotonic() - started > timeout:
                        process.kill()
                        record["timed_out"] = True
                        break
                    time.sleep(0.02)
                process.wait()
                record["returncode"] = None if record.get("timed_out") else process.returncode
            record["window_sent_to_back"] = watch.moved_back
            record["window_foreground_samples"] = watch.foreground_samples
            record["window_samples"] = watch.samples
            record["wall_s"] = time.monotonic() - started
            record["host_after"] = host_notes()
            record["summary"] = summarize(run_dir, plan)
            (run_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            if plan.get("window_phase"):
                measured = (
                    f"window={_short(record['summary'].get('window'))} "
                    f"invalid={record['summary'].get('window_invalid')}"
                )
            else:
                measured = f"steady={_short(record['summary'].get('steady'))}"
            print(
                f"{run_id}: rc={record.get('returncode')} wall={record['wall_s']:.1f}s "
                f"frames={record['summary'].get('frames')} "
                f"settled={record['summary'].get('settled_frame')} "
                f"focus={watch.foreground_samples}/{watch.samples} "
                f"{measured}",
                flush=True,
            )
            # Raw traces are kept compressed. Game directories are disposable.
            shutil.rmtree(game_dir, ignore_errors=True)
            for name in ("frames.csv", "frames.csv.phases.csv", "frames.csv.work.csv", SIDECAR_NAME):
                raw = run_dir / name
                if raw.exists():
                    with raw.open("rb") as source, gzip.open(str(raw) + ".gz", "wb", 6) as target:
                        shutil.copyfileobj(source, target)
                    raw.unlink()
    analyze(output)
    return 0


def warn_relative_args(args: list[str]) -> list[str]:
    """Plan `args` that name a file relative to this directory: the client, which runs in the
    run directory, would not find them. Warns on stderr and returns them; changes nothing."""
    found = [arg for arg in args if not Path(arg).is_absolute() and Path(arg).exists()]
    for arg in found:
        print(
            f"warning: plan arg {arg!r} is a relative path; the client runs in its run directory "
            f"and will not find it. Use {str(Path(arg).resolve())!r}.",
            file=sys.stderr,
            flush=True,
        )
    return found


def _short(window: object) -> str:
    if not isinstance(window, dict):
        return "none"
    return f"mean {window['mean']:.3f} p99 {window['p99']:.3f}"


def paired(control: list[float], candidate: list[float]) -> dict[str, float]:
    """Candidate-minus-control over paired runs, with a t interval at the run level."""
    differences = [b - a for a, b in zip(control, candidate)]
    n = len(differences)
    mean = statistics.fmean(differences)
    result = {
        "n_pairs": n,
        "control_mean": statistics.fmean(control),
        "candidate_mean": statistics.fmean(candidate),
        "difference": mean,
        "percent": 100.0 * mean / statistics.fmean(control) if statistics.fmean(control) else math.nan,
        "differences": differences,
    }
    if n >= 2:
        half = T_975.get(n - 1, 2.0) * statistics.stdev(differences) / math.sqrt(n)
        result["ci95"] = [mean - half, mean + half]
    return result


def _missing_last(value: object) -> tuple[bool, object]:
    """Sort key: integers in their usual order, a missing (None) value last."""
    return (value is None, 0 if value is None else value)


def window_sum_gate(records: dict, complete: list[int]) -> dict:
    """The per-run window sums of `window_sum_counters` must be identical across runs."""
    values: dict[str, set] = {}
    missing: list[str] = []
    for pair in complete:
        for lane in "AB":
            record = records[(pair, lane)]
            sums = record["summary"].get("window_sums") or {}
            if not sums or None in sums.values():
                missing.append(record["run"])
            for key, value in sums.items():
                values.setdefault(key, set()).add(value)
    return {
        "identical": bool(complete) and not missing and all(len(v) == 1 for v in values.values()),
        "values": {key: sorted(v, key=_missing_last) for key, v in values.items()},
        "runs_missing": missing,
    }


def sidecar_equality_gate(output: Path, records: dict, complete: list[int], plan: dict) -> dict | None:
    """Every complete run's sidecar rows must be identical; None when no run has a sidecar."""
    digests = {
        records[(pair, lane)]["run"]: records[(pair, lane)]["summary"].get("sidecar")
        for pair in complete
        for lane in "AB"
    }
    if not any(digests.values()):
        return None
    missing = [run for run, digest in digests.items() if digest is None]
    distinct = sorted({(digest["rows"], digest["sha256"]) for digest in digests.values() if digest})
    gate: dict[str, object] = {
        "identical": bool(complete) and not missing and len(distinct) == 1,
        "rows": sorted({rows for rows, _ in distinct}),
        "distinct_digests": len(distinct),
        "runs_missing": missing,
    }
    if len(distinct) > 1:
        gate["first_divergence"] = first_sidecar_divergence(output, digests, plan)
    return gate


def first_sidecar_divergence(output: Path, digests: dict, plan: dict) -> dict:
    """The first compared cell where some run's sidecar departs from the first run's."""
    present = [(run, digest) for run, digest in digests.items() if digest]
    reference_run, reference = present[0]
    other_run = next(
        run
        for run, digest in present
        if (digest["rows"], digest["sha256"]) != (reference["rows"], reference["sha256"])
    )
    rows_a = read_sidecar(output / reference_run) or []
    rows_b = read_sidecar(output / other_run) or []
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        for column in sidecar_columns(rows_a, plan.get("sidecar_ignore", ["frame"])):
            if row_a.get(column) != row_b.get(column):
                return {
                    "reference_run": reference_run,
                    "run": other_run,
                    "row": index,
                    "step": row_a.get("step"),
                    "column": column,
                    "reference": row_a.get(column),
                    "value": row_b.get(column),
                }
    return {
        "reference_run": reference_run,
        "run": other_run,
        "row": min(len(rows_a), len(rows_b)),
        "reference_rows": len(rows_a),
        "rows": len(rows_b),
    }


def analyze(output: Path) -> dict:
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    records = {}
    for path in sorted(output.glob("p*-?/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        # Recomputing keeps analysis independent of what the live loop printed.
        record["summary"] = summarize(path.parent, plan)
        records[(record["pair"], record["lane"])] = record

    window_mode = bool(plan.get("window_phase"))

    def valid(record: dict) -> bool:
        if window_mode and "window_invalid" in record["summary"]:
            return False
        return record.get("returncode") == 0 and record["summary"].get("frames", 0) > 0

    pairs = sorted({pair for pair, _ in records})
    invalid = sorted(r["run"] for r in records.values() if not valid(r))
    complete = [p for p in pairs if valid(records.get((p, "A"), {"summary": {}})) and valid(records.get((p, "B"), {"summary": {}}))]

    def series(lane: str, window: str, metric: str) -> list[float] | None:
        values = []
        for pair in complete:
            block = records[(pair, lane)]["summary"].get(window)
            if not isinstance(block, dict) or metric not in block:
                return None
            values.append(float(block[metric]))
        return values

    comparisons: dict[str, dict] = {}
    # Only window plans produce a "window" block, so other plans compare exactly as before.
    for window in ("steady", "window", "load"):
        for metric in ("mean", "p50", "p95", "p99", "p999", "max"):
            control = series("A", window, metric)
            candidate = series("B", window, metric)
            if control and candidate:
                comparisons[f"{window}.{metric}"] = paired(control, candidate)
    control = [float(records[(p, "A")]["summary"].get("load_ms", math.nan)) for p in complete]
    candidate = [float(records[(p, "B")]["summary"].get("load_ms", math.nan)) for p in complete]
    if control and not any(math.isnan(v) for v in control + candidate):
        comparisons["load.total_ms"] = paired(control, candidate)
    names = set()
    for record in records.values():
        names.update(record["summary"].get("steady_phases", {}))
    for name in sorted(names):
        try:
            control = [float(records[(p, "A")]["summary"]["steady_phases"][name]) for p in complete]
            candidate = [float(records[(p, "B")]["summary"]["steady_phases"][name]) for p in complete]
        except KeyError:
            continue
        comparisons[f"steady_phase.{name}"] = paired(control, candidate)
    slice_comparisons: list[dict] = []
    if window_mode:
        names = set()
        for record in records.values():
            names.update(record["summary"].get("window_phases", {}))
        for name in sorted(names):
            try:
                control = [float(records[(p, "A")]["summary"]["window_phases"][name]) for p in complete]
                candidate = [float(records[(p, "B")]["summary"]["window_phases"][name]) for p in complete]
            except KeyError:
                continue
            comparisons[f"window_phase.{name}"] = paired(control, candidate)
        slice_counts = {
            len(records[(p, lane)]["summary"].get("window_slices", [])) for p in complete for lane in "AB"
        }
        if len(slice_counts) == 1:
            for index in range(slice_counts.pop()):
                slice_comparisons.append(
                    {
                        metric: paired(
                            [float(records[(p, "A")]["summary"]["window_slices"][index][metric]) for p in complete],
                            [float(records[(p, "B")]["summary"]["window_slices"][index][metric]) for p in complete],
                        )
                        for metric in METRICS
                    }
                )

    # Correctness gate: the settled scene has to be the same scene in every run. A run with no
    # final_work has nothing to compare, and neither has a batch with no complete pairs: both
    # fail the gate. (They used to pass it: `all([])` is True.)
    work_values: dict[str, set] = {}
    missing_final_work: list[str] = []
    for pair in complete:
        for lane in "AB":
            record = records[(pair, lane)]
            final_work = record["summary"].get("final_work") or {}
            if not final_work or None in final_work.values():
                missing_final_work.append(record["run"])
            for key, value in final_work.items():
                work_values.setdefault(key, set()).add(value)
    work_gate = {key: sorted(values, key=_missing_last) for key, values in work_values.items()}
    report = {
        "batch": plan["batch"],
        "lane_env": plan.get("lane_env", {}),
        "identical_executables": json.loads((output / "identity.json").read_text())["A"]
        == json.loads((output / "identity.json").read_text())["B"],
        "pairs_complete": complete,
        "invalid_runs": invalid,
        "settled_work_values": work_gate,
        "settled_work_identical": bool(complete)
        and not missing_final_work
        and all(len(v) == 1 for v in work_gate.values()),
        "comparisons": comparisons,
        "per_run": {
            record["run"]: {
                "returncode": record.get("returncode"),
                "wall_s": record.get("wall_s"),
                **{k: v for k, v in record["summary"].items() if k != "final_work"},
            }
            for record in records.values()
        },
    }
    # Only present when the gate had nothing to check, so complete batches keep their old shape.
    if not complete or missing_final_work:
        report["settled_work_unchecked"] = {
            "no_complete_pairs": not complete,
            "runs_without_final_work": missing_final_work,
        }
    if window_mode:
        report["window"] = {
            "phase": plan["window_phase"],
            "frames_expected": plan.get("window_frames_expected"),
            "tail_phase": plan.get("tail_phase"),
            "slices": len(slice_comparisons),
        }
        report["window_slice_comparisons"] = slice_comparisons
        gates: dict[str, dict] = {
            "final_work": {
                "identical": report["settled_work_identical"],
                "counters": sorted(work_gate),
            }
        }
        if plan.get("window_sum_counters"):
            gates["window_sums"] = window_sum_gate(records, complete)
        sidecar_gate = sidecar_equality_gate(output, records, complete, plan)
        if sidecar_gate is not None:
            gates["sidecar"] = sidecar_gate
        report["same_work"] = gates
        report["same_work_identical"] = all(gate["identical"] for gate in gates.values())
    (output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "analysis.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return report


def render_markdown(report: dict) -> str:
    lines = [f"# {report['batch']}", ""]
    kind = "A/A noise baseline" if report["identical_executables"] and not report.get("lane_env") else "control (A) vs candidate (B)"
    lines.append(
        f"{kind}. {len(report['pairs_complete'])} complete pairs. "
        f"Invalid runs: {report['invalid_runs'] or 'none'}. "
        f"Settled work identical: {report['settled_work_identical']}."
    )
    lines.append("")
    lines.append("| metric (ms) | A | B | B-A | % | 95% CI of B-A |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, block in report["comparisons"].items():
        ci = block.get("ci95")
        ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a"
        lines.append(
            f"| {name} | {block['control_mean']:.4f} | {block['candidate_mean']:.4f} | "
            f"{block['difference']:+.4f} | {block['percent']:+.2f} | {ci_text} |"
        )
    unchecked = report.get("settled_work_unchecked")
    if unchecked:
        reasons = []
        if unchecked["no_complete_pairs"]:
            reasons.append("there are no complete pairs")
        if unchecked["runs_without_final_work"]:
            reasons.append(
                "runs " + ", ".join(unchecked["runs_without_final_work"]) + " have no final_work "
                "(never settled, so no steady window to read it from; or a counter was empty)"
            )
        because = "; ".join(reasons)
        lines.append("")
        lines.append(
            "**Same-work gate FAILED: nothing was compared.** " + because[:1].upper() + because[1:] + ". "
            "Before 2026-11-05 this case reported `Settled work identical: True`, because `all()` "
            "over no values is True; an empty check is now a failure, not a pass."
        )
    differing = {key: values for key, values in report["settled_work_values"].items() if len(values) > 1}
    if not report["settled_work_identical"] and differing:
        lines.append("")
        lines.append("Settled work differed between runs:")
        for key, values in differing.items():
            lines.append(f"- {key}: {values[:8]}")
    if report.get("window"):
        render_window_markdown(report, lines)
    lines.append("")
    return "\n".join(lines)


def render_window_markdown(report: dict, lines: list[str]) -> None:
    window = report["window"]
    lines.append("")
    detail = f"Window: the frames labelled `{window['phase']}`"
    if window.get("frames_expected") is not None:
        detail += f", {window['frames_expected']} expected"
    if window.get("tail_phase"):
        detail += f"; final_work is read on the last frame of `{window['tail_phase']}`"
    lines.append(detail + ".")
    verdicts = ", ".join(
        f"{name} {'identical' if gate['identical'] else 'NOT identical'}"
        for name, gate in report["same_work"].items()
    )
    lines.append(f"Same work: {'PASS' if report['same_work_identical'] else 'FAIL'} ({verdicts}).")
    divergence = report["same_work"].get("sidecar", {}).get("first_divergence")
    if divergence:
        lines.append(f"First sidecar divergence: `{json.dumps(divergence)}`")
    lines.append("")
    lines.append("| run | window frames | busy fraction | settled frame | invalid |")
    lines.append("|---|---:|---:|---:|---|")
    for run, info in sorted(report["per_run"].items()):
        busy = info.get("window_busy_fraction")
        busy_text = f"{busy:.4f}" if isinstance(busy, float) else "n/a"
        lines.append(
            f"| {run} | {info.get('window_frames', 'n/a')} | {busy_text} | "
            f"{info.get('settled_frame')} | {info.get('window_invalid') or ''} |"
        )
    slices = report.get("window_slice_comparisons") or []
    if slices:
        lines.append("")
        lines.append("| slice | mean A | mean B | B-A | % | p99 A | p99 B | p99 B-A |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for index, block in enumerate(slices):
            mean, p99 = block["mean"], block["p99"]
            lines.append(
                f"| {index} | {mean['control_mean']:.4f} | {mean['candidate_mean']:.4f} | "
                f"{mean['difference']:+.4f} | {mean['percent']:+.2f} | {p99['control_mean']:.4f} | "
                f"{p99['candidate_mean']:.4f} | {p99['difference']:+.4f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    again = commands.add_parser("analyze")
    again.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        return run_batch(args.plan, args.output)
    analyze(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
