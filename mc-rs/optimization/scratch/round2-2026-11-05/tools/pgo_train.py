"""PGO training: run instrumented binaries once per workload with each plan's exact options and args.

    python -I pgo_train.py OPTIMIZATION_DIR INSTRUMENTED_DIST_DIR OUT_DIR

INSTRUMENTED_DIST_DIR holds mc-client.exe and examples/skywars_bench.exe built with
-Cprofile-generate; their .profraw files land where that flag points. Workloads mirror the A/B
scenes: SkyWars (lockstep bench, pipelined submit as installed), dense terrain fixed and orbiting
camera, the e1-16 entity scene. Training runs are never timed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from quick_ab import BASE_OPTIONS, checkout_jars_dir  # noqa: E402

dist = Path(sys.argv[2])
out = Path(sys.argv[3])
plans = Path(__file__).resolve().parent.parent / "plans"
# The plans name repository files as {MC_RS}\...: OPTIMIZATION_DIR is <mc-rs>\optimization, so the
# workspace root comes from it and the repository can be cloned anywhere.
mc_rs = Path(sys.argv[1]).resolve().parent


def expand(value: str) -> str:
    return value.replace("{MC_RS}", str(mc_rs))


WORKLOADS = [
    # (name, plan, exe under dist, extra env)
    ("skywars-pipelined", "pm-latency-sw.json", "examples/skywars_bench.exe", {"MC_PIPELINED_SUBMIT": "1"}),
    ("skywars-serial", "pm-latency-sw.json", "examples/skywars_bench.exe", {}),
    ("terrain-fixed", "lunch-pipe-terrain-fixed.json", "mc-client.exe", {"MC_PIPELINED_SUBMIT": "1"}),
    ("terrain-orbit", "lunch-pipe-terrain-orbit.json", "mc-client.exe", {"MC_PIPELINED_SUBMIT": "1"}),
    ("e1-16", "lunch-pipe-e1-16.json", "mc-client.exe", {"MC_PIPELINED_SUBMIT": "1"}),
]

out.mkdir(parents=True, exist_ok=False)
for name, plan_name, exe, extra in WORKLOADS:
    plan = json.loads((plans / plan_name).read_text(encoding="utf-8"))
    run = out / name
    game = run / "game"
    game.mkdir(parents=True)
    options = {**BASE_OPTIONS, **plan.get("options", {})}
    (game / "options.txt").write_text("".join(f"{k}:{v}\n" for k, v in options.items()), encoding="utf-8")
    env = dict(os.environ)
    # The plans pass no --versionJar and an optimised build has no checkout path compiled in.
    jars = checkout_jars_dir()
    if jars is not None:
        env.setdefault("MC_JARS_DIR", jars)
    # Training runs are offline and deterministic: the client starts no skin download worker
    # (skywars_bench never does).
    env.setdefault("MC_NO_SKIN_DOWNLOADS", "1")
    env.update({k: expand(v) for k, v in plan.get("env", {}).items()})
    env.update({"MC_BACKGROUND_WINDOW": "1", "MC_FRAME_TRACE": str((run / "frames.csv").resolve())})
    env.update(extra)
    args = [expand(a) for a in plan["args"]]
    for flag, value in zip(args, args[1:]):
        if flag in ("--replay", "--script") and not Path(value).is_file():
            sys.exit(f"{name}: the {flag} file is missing: {value}")
    cmd = [str(dist / exe), *args, "--gameDir", str(game.resolve())]
    with open(run / "stdout.log", "w", encoding="utf-8") as so, open(run / "stderr.log", "w", encoding="utf-8") as se:
        code = subprocess.run(cmd, cwd=str(run), env=env, stdout=so, stderr=se, timeout=3600).returncode
    tail = (run / "stderr.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-2:]
    print(f"{name}: exit {code}; {' | '.join(tail)[:200]}", flush=True)
    if code != 0:
        sys.exit(code)
