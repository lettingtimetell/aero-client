import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import benchmark_lease
from benchmark_lease import BenchmarkLease, queue_status


class BenchmarkLeaseTests(unittest.TestCase):
    class FailingLease(BenchmarkLease):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail_writes = True

        def _write_metadata(self, status="held"):
            if self.fail_writes:
                raise OSError("injected metadata failure")
            return super()._write_metadata(status)

    def test_context_metadata_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            with BenchmarkLease(path, timeout=0.2, heartbeat_interval=0.02, owner="test") as lease:
                self.assertEqual(lease.metadata["owner"], "test")
                self.assertEqual(lease.metadata["pid"], os.getpid())
                self.assertTrue(lease.metadata_path.exists())
                self.assertEqual(json.loads(lease.metadata_path.read_text())["run_id"], lease.run_id)
            self.assertEqual(json.loads(lease.metadata_path.read_text())["status"], "released")

    def test_subprocess_contention_then_release_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            script = (
                "import sys,time; sys.path.insert(0, sys.argv[2]); "
                "from benchmark_lease import BenchmarkLease; "
                "x=BenchmarkLease(sys.argv[1], timeout=2); x.acquire(); print('held',flush=True); time.sleep(.35)"
            )
            child = subprocess.Popen([sys.executable, "-c", script, str(path), str(Path(__file__).parent)], stdout=subprocess.PIPE, text=True)
            self.assertEqual(child.stdout.readline().strip(), "held")
            with self.assertRaises(TimeoutError):
                BenchmarkLease(path, timeout=0.08).acquire()
            child.kill()
            self.assertNotEqual(child.wait(timeout=3), 0)
            child.stdout.close()
            with BenchmarkLease(path, timeout=0.5):
                pass

    def test_background_heartbeat_updates_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            with BenchmarkLease(path, timeout=0.2, heartbeat_interval=0.02):
                first = json.loads(Path(str(path) + ".json").read_text())["heartbeat"]
                time.sleep(0.08)
                second = json.loads(Path(str(path) + ".json").read_text())["heartbeat"]
                self.assertGreater(second, first)

    def test_metadata_failure_on_acquire_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            with self.assertRaises(OSError):
                self.FailingLease(path, timeout=0.2).acquire()
            with BenchmarkLease(path, timeout=0.2):
                pass

    def test_metadata_failure_on_release_still_unlocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            lease = self.FailingLease(path, timeout=0.2)
            lease.fail_writes = False
            lease.acquire()
            lease.fail_writes = True
            with self.assertRaises(OSError):
                lease.release()
            with BenchmarkLease(path, timeout=0.2):
                pass

    def test_stale_metadata_does_not_steal_live_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            with BenchmarkLease(path, timeout=0.1):
                data = json.loads(Path(str(path) + ".json").read_text())
                data["heartbeat"] = 0
                Path(str(path) + ".json").write_text(json.dumps(data))
                with self.assertRaises(TimeoutError):
                    BenchmarkLease(path, timeout=0.05).acquire()

    def test_reacquire_refreshes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            lease = BenchmarkLease(Path(directory) / "lease.lock", timeout=0.2)
            lease.acquire()
            first = lease.run_id
            lease.release()
            lease.acquire()
            self.assertNotEqual(first, lease.run_id)
            lease.release()

    def test_failed_background_heartbeat_is_observable_without_losing_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            lease = self.FailingLease(path, timeout=0.1, heartbeat_interval=0.01)
            lease.fail_writes = False
            lease.acquire()
            try:
                lease.fail_writes = True
                self.assertTrue(lease._stop.wait(2), "failed heartbeat must report its error")
                with self.assertRaisesRegex(RuntimeError, "heartbeat failed"):
                    lease.check_health()
                with self.assertRaises(TimeoutError):
                    BenchmarkLease(path, timeout=0).acquire()
            finally:
                lease.fail_writes = False
                lease.release()
            with self.assertRaisesRegex(RuntimeError, "not held"):
                lease.heartbeat()

    def test_nonfinite_timeouts_are_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BenchmarkLease(timeout=value)

    def test_default_path_is_cwd_independent(self):
        code = "import os; from benchmark_lease import BenchmarkLease; print(BenchmarkLease().lock_path)"
        values = []
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).parent)
        for cwd in (tempfile.gettempdir(), str(Path(__file__).parent)):
            values.append(subprocess.check_output([sys.executable, "-c", code], cwd=cwd, env=environment, text=True).strip())
        self.assertEqual(values[0], values[1])


# A waiter process: argv = lock path, module folder, log file, name, seconds to hold the lease.
WAITER = (
    "import sys,time; sys.path.insert(0, sys.argv[2]); "
    "from benchmark_lease import BenchmarkLease; "
    "x=BenchmarkLease(sys.argv[1], timeout=60, owner=sys.argv[4]); x.acquire(); "
    "open(sys.argv[3],'a').write(sys.argv[4]+'\\n'); time.sleep(float(sys.argv[5])); x.release()"
)


def tickets(path):
    queue = Path(str(path) + benchmark_lease.QUEUE_SUFFIX)
    if not queue.exists():
        return []
    return sorted(name for name in os.listdir(queue) if name.startswith(benchmark_lease.TICKET_PREFIX))


class FirstComeFirstServedTests(unittest.TestCase):
    """Waiters get the lease in the order they started waiting."""

    def start_waiter(self, path, log, name, hold=0.1):
        return subprocess.Popen(
            [sys.executable, "-c", WAITER, str(path), str(Path(__file__).parent), str(log), name, str(hold)]
        )

    def wait_for_tickets(self, path, count, timeout=60):
        deadline = time.monotonic() + timeout
        while len(tickets(path)) < count:
            if time.monotonic() > deadline:
                self.fail(f"{count} tickets never appeared: {tickets(path)}")
            time.sleep(0.01)

    def test_waiters_are_served_in_arrival_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            log = Path(directory) / "order.log"
            names = ["w1", "w2", "w3", "w4", "w5"]
            children = []
            with BenchmarkLease(path, timeout=0.2):
                for count, name in enumerate(names, 1):
                    children.append(self.start_waiter(path, log, name))
                    self.wait_for_tickets(path, count)
            for child in children:
                self.assertEqual(child.wait(timeout=120), 0)
            self.assertEqual(log.read_text().split(), names)
            self.assertEqual(tickets(path), [])

    def test_a_newcomer_cannot_overtake_a_queued_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            log = Path(directory) / "order.log"
            holder = BenchmarkLease(path, timeout=0.2)
            holder.acquire()
            child = self.start_waiter(path, log, "queued", hold=0.3)
            try:
                self.wait_for_tickets(path, 1)
                holder.release()
                # The lock is free for a moment, but the queued waiter is first in line. (With
                # the old unordered polling this newcomer usually got the lease here.)
                with self.assertRaises(TimeoutError):
                    BenchmarkLease(path, timeout=0).acquire()
            finally:
                self.assertEqual(child.wait(timeout=120), 0)
            self.assertEqual(log.read_text().split(), ["queued"])

    def test_a_crashed_waiter_is_skipped_and_its_ticket_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            log = Path(directory) / "order.log"
            with BenchmarkLease(path, timeout=0.2):
                child = self.start_waiter(path, log, "crashed")
                self.wait_for_tickets(path, 1)
                child.kill()
                child.wait(timeout=30)
            self.assertEqual(len(tickets(path)), 1, "a killed waiter leaves its ticket file")
            with BenchmarkLease(path, timeout=2) as lease:
                self.assertEqual(lease.metadata["queue_ahead_at_entry"], 1)
                self.assertEqual(lease.metadata["queue"], "fifo")
            self.assertEqual(tickets(path), [])
            self.assertFalse(log.exists())

    def test_a_waiter_that_gives_up_leaves_the_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            with BenchmarkLease(path, timeout=0.2):
                with self.assertRaises(TimeoutError):
                    BenchmarkLease(path, timeout=0.3).acquire()
                self.assertEqual(tickets(path), [])

    def test_a_suspended_waiter_is_skipped_but_a_live_one_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            sleeper = BenchmarkLease(path, owner="suspended")
            self.assertEqual(sleeper._enqueue(), 0)  # holds a live ticket and never polls
            try:
                # Nobody holds the lock, but a live waiter is ahead: wait your turn.
                with self.assertRaises(TimeoutError):
                    BenchmarkLease(path, timeout=0.3).acquire()
                stale = time.time() - benchmark_lease.QUEUE_STALE_S - 5
                sleeper._refresh_ticket(force=True, heartbeat=stale)
                with BenchmarkLease(path, timeout=2) as lease:
                    self.assertEqual(lease.metadata["queue_ahead_at_entry"], 1)
                self.assertEqual(len(tickets(path)), 1, "a stale ticket is skipped, not removed")
            finally:
                sleeper._dequeue()
            self.assertEqual(tickets(path), [])

    def test_ticket_numbers_are_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            lease = BenchmarkLease(Path(directory) / "lease.lock")
            numbers = []
            for _ in range(3):
                lease._enqueue()
                numbers.append(lease._ticket)
                lease._dequeue()
            self.assertEqual(numbers, sorted(set(numbers)))
            self.assertEqual(tickets(lease.lock_path), [])

    def test_status_lists_the_holder_and_the_queue_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.lock"
            first = BenchmarkLease(path, owner="first-waiter")
            second = BenchmarkLease(path, owner="second-waiter")
            with BenchmarkLease(path, timeout=0.2, owner="the-holder"):
                first._enqueue()
                second._enqueue()
                try:
                    report = queue_status(path)
                finally:
                    first._dequeue()
                    second._dequeue()
            self.assertEqual(report["holder"]["owner"], "the-holder")
            self.assertEqual(report["holder"]["status"], "held")
            self.assertEqual([t["owner"] for t in report["queue"]], ["first-waiter", "second-waiter"])
            self.assertEqual([t["state"] for t in report["queue"]], ["live", "live"])
            text = benchmark_lease._describe(report)
            self.assertIn("held by the-holder", text)
            self.assertLess(text.index("first-waiter"), text.index("second-waiter"))
            self.assertEqual(queue_status(path)["queue"], [])


if __name__ == "__main__":
    unittest.main()
