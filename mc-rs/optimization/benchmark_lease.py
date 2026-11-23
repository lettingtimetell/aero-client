"""Small cross-process lease used to serialize host benchmarks.

The lease is an advisory OS lock.  A timeout only bounds waiting: it never
deletes, replaces, or steals a live lock.  The lock inode/file is deliberately
kept in place so every contender addresses the same kernel lock object.

Waiters are served first come, first served. Before, every waiter
polled the lock and whoever polled first after a release won, so a batch that had waited an hour
could lose to one that arrived a second earlier. Now a waiter takes a numbered ticket in the
folder ``<lock>.queue`` and holds an OS lock on its ticket file while it waits, and only the
waiter with the lowest live ticket tries the lock. The queue orders the attempts; the kernel lock
alone still decides who holds the lease, so every guarantee above is unchanged:

* a waiter that crashes loses its ticket lock with its process, so the next waiter skips that
  ticket and removes it;
* a waiter that is alive but has not refreshed its ticket for ``QUEUE_STALE_S`` seconds (a
  suspended process) is skipped too, so it cannot hold the queue up; it gets its place back when
  it refreshes again;
* a waiter that times out or is interrupted removes its own ticket;
* if the queue folder cannot be used, the lease falls back to the old unordered polling and says
  so on stderr.

A process that uses an older copy of this module takes no ticket and still competes in the old
way. ``python benchmark_lease.py status`` prints the holder and the queue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

if os.name == "nt":
    import msvcrt
else:
    import fcntl

# ---- the first-come-first-served queue ----
QUEUE_SUFFIX = ".queue"
# The numbering lock, held only while a ticket is numbered, created and locked. Its content is
# the last ticket number handed out, so numbers are never reused.
REGISTRY_NAME = "registry.lock"
TICKET_PREFIX = "ticket-"
TICKET_SUFFIX = ".json"
# Windows byte-range locks are mandatory (a locked byte cannot be read through another handle),
# so queue files lock one byte far past their small text body, which stays readable.
QUEUE_LOCK_OFFSET = 1 << 20
# Each ticket body is padded to this size and rewritten in place, so a reader never sees it
# change length.
TICKET_BODY_BYTES = 512
# A waiting process rewrites its ticket's heartbeat this often...
TICKET_REFRESH_S = 1.0
# ...and a live ticket whose heartbeat is older than this is skipped (but not removed).
QUEUE_STALE_S = 60.0
# How long a waiter may wait for the numbering lock before it falls back to unordered polling.
REGISTRY_WAIT_S = 10.0
# The first waiter in line polls the lease lock this often; the others watch the queue.
HEAD_POLL_S = 0.05
QUEUE_POLL_S = 0.25


def _queue_lock(stream: Any) -> bool:
    """Try to lock a queue file (one byte at ``QUEUE_LOCK_OFFSET`` on Windows); never blocks."""
    try:
        if os.name == "nt":
            stream.seek(QUEUE_LOCK_OFFSET)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (OSError, BlockingIOError):
        return False


def _queue_unlock(stream: Any) -> None:
    if os.name == "nt":
        stream.seek(QUEUE_LOCK_OFFSET)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _open_read_write(path: Path) -> Any:
    """Open (creating if needed) for reading and writing at any offset; never truncates."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o666)
    return os.fdopen(fd, "r+b")


def _ticket_number(name: str) -> Optional[int]:
    if not (name.startswith(TICKET_PREFIX) and name.endswith(TICKET_SUFFIX)):
        return None
    digits = name[len(TICKET_PREFIX) : -len(TICKET_SUFFIX)]
    return int(digits) if digits.isdigit() else None


def _list_tickets(queue_dir: Path) -> list[tuple[int, Path]]:
    """Every ticket file in the queue, lowest number first."""
    try:
        names = os.listdir(queue_dir)
    except OSError:
        return []
    found = []
    for name in names:
        number = _ticket_number(name)
        if number is not None:
            found.append((number, queue_dir / name))
    found.sort()
    return found


def _read_ticket(path: Path) -> dict[str, Any]:
    """A ticket's body; empty when it is gone or caught mid-rewrite."""
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _probe_ticket(path: Path) -> str:
    """``"live"`` while its waiter holds the ticket lock, ``"dead"`` when nobody does, or
    ``"gone"``. A ticket that cannot be opened for another reason counts as live."""
    try:
        stream = open(path, "rb")
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "live"
    try:
        if not _queue_lock(stream):
            return "live"
        _queue_unlock(stream)
        return "dead"
    finally:
        stream.close()


def _remove_quietly(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _default_lock_path() -> Path:
    """Return a host-wide path which does not depend on the current directory."""
    if os.name == "nt":
        program_data = os.environ.get("PROGRAMDATA")
        if not program_data:
            raise RuntimeError("PROGRAMDATA is unavailable; cannot choose a machine-wide lease path")
        return Path(program_data) / "mc-rs" / "benchmark.lock"
    return Path("/tmp/mc-rs-benchmark.lock")


def _process_start_identity() -> str:
    """Return a best-effort identity which distinguishes PID reuse."""
    if os.name != "nt":
        try:
            stat = Path(f"/proc/{os.getpid()}/stat").read_text()
            # comm may contain spaces; the final ')' terminates it.  The
            # remainder starts at field 3, making starttime (field 22) index 19.
            fields = stat[stat.rfind(")") + 2 :].split()
            return f"proc-start:{fields[19]}"
        except (OSError, IndexError):
            pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            ok = kernel32.GetProcessTimes(
                kernel32.GetCurrentProcess(), ctypes.byref(creation), ctypes.byref(exit_time),
                ctypes.byref(kernel), ctypes.byref(user)
            )
            if ok:
                value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                return f"win-creation:{value}"
        except (AttributeError, OSError) as exc:
            raise RuntimeError("cannot obtain Windows process creation identity") from exc
    raise RuntimeError("cannot obtain process start identity")


class BenchmarkLease:
    """An exclusive, OS-backed host lease for a benchmark run.

    ``timeout`` controls how long acquisition waits.  The kernel lock is the
    authority; metadata is informational and may be stale after a crash.
    Process exit closes the descriptor and lets Windows/POSIX recover an
    abandoned lease automatically.  The lock file is never unlinked. Acquire
    and release belong to one coordinator thread; heartbeat calls may come
    from another thread. Do not share an acquired instance across a fork.
    Waiters are served in arrival order (see the module docstring); a waiter
    that gives up loses its place.
    """

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        *,
        timeout: float = 30.0,
        heartbeat_interval: float = 5.0,
        owner: Optional[str] = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout < 0 or not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0:
            raise ValueError("timeout must be non-negative and heartbeat_interval positive")
        self.lock_path = (Path(path) if path is not None else _default_lock_path()).resolve()
        self.metadata_path = Path(str(self.lock_path) + ".json")
        self.timeout = float(timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self.owner = owner or f"pid-{os.getpid()}"
        self.run_id = secrets.token_hex(16)
        self.process_start_identity = _process_start_identity()
        self._file: Any = None
        self._held = False
        self._stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._metadata: dict[str, Any] = {}
        self._metadata_lock = threading.RLock()
        self._heartbeat_error: Optional[BaseException] = None
        # The waiting ticket: number, open locked file, path.
        self.queue_dir = Path(str(self.lock_path) + QUEUE_SUFFIX)
        self._ticket: Optional[int] = None
        self._ticket_stream: Any = None
        self._ticket_path: Optional[Path] = None
        self._ticket_refreshed = 0.0
        self._ticket_enqueued = 0.0
        self._queue_note = "fifo"

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of the current lease metadata."""
        with self._metadata_lock:
            return dict(self._metadata)

    def _try_lock(self) -> bool:
        assert self._file is not None
        try:
            if os.name == "nt":
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            return False

    # ---- the first-come-first-served queue ----

    def _enqueue(self) -> int:
        """Take the next ticket and hold its lock; return how many tickets were already queued.

        Numbering, creating and locking a ticket all happen under the registry lock, and a waiter
        only ever looks at tickets numbered below its own, so nobody can see a new ticket before
        it is locked (it would look dead). Without a usable queue folder this waiter polls
        unordered, as every waiter did before the queue existed.
        """
        self._ticket = None
        self._queue_note = "fifo"
        try:
            self.queue_dir.mkdir(parents=True, exist_ok=True)
            registry = _open_read_write(self.queue_dir / REGISTRY_NAME)
        except OSError as error:
            return self._queue_unavailable(f"cannot open the queue ({error})")
        try:
            give_up = time.monotonic() + REGISTRY_WAIT_S
            while not _queue_lock(registry):
                if time.monotonic() >= give_up:
                    return self._queue_unavailable("the queue registry stayed locked")
                time.sleep(0.002)
            try:
                return self._take_ticket(registry)
            except OSError as error:
                return self._queue_unavailable(f"cannot take a ticket ({error})")
            finally:
                _queue_unlock(registry)
        finally:
            registry.close()

    def _take_ticket(self, registry: Any) -> int:
        """Under the registry lock: create, lock and publish the next ticket."""
        tickets = _list_tickets(self.queue_dir)
        registry.seek(0)
        raw = registry.read(32).decode("ascii", "replace").strip()
        last = int(raw) if raw.isdigit() else 0
        number = max([last] + [queued for queued, _ in tickets]) + 1
        while True:
            path = self.queue_dir / f"{TICKET_PREFIX}{number:012d}{TICKET_SUFFIX}"
            try:
                flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
                fd = os.open(path, flags, 0o666)
                break
            except FileExistsError:
                number += 1
        stream = os.fdopen(fd, "r+b")
        try:
            if not _queue_lock(stream):
                raise OSError(f"cannot lock the new ticket {path}")
            self._ticket_stream, self._ticket_path, self._ticket = stream, path, number
            self._ticket_enqueued = time.time()
            self._refresh_ticket(force=True)
            registry.seek(0)
            registry.write(f"{number:020d}".encode("ascii"))
            registry.flush()
        except BaseException:
            self._ticket_stream = self._ticket_path = self._ticket = None
            stream.close()
            _remove_quietly(path)
            raise
        return len(tickets)

    def _queue_unavailable(self, why: str) -> int:
        self._dequeue()
        self._ticket = None
        self._queue_note = f"unavailable: {why}"
        print(f"benchmark lease: {why}; waiting without a place in the queue", file=sys.stderr)
        return 0

    def _refresh_ticket(self, *, force: bool = False, heartbeat: Optional[float] = None) -> None:
        """Rewrite this waiter's ticket body at most every ``TICKET_REFRESH_S`` seconds.

        ``heartbeat`` replaces the time written (tests use it to age a ticket). A failed write
        is ignored: the ticket then goes stale and is skipped, which only costs its place.
        """
        stream = self._ticket_stream
        if stream is None:
            return
        now = time.monotonic()
        if not force and now - self._ticket_refreshed < TICKET_REFRESH_S:
            return
        self._ticket_refreshed = now
        body = json.dumps(
            {
                "ticket": self._ticket,
                "owner": self.owner[:200],
                "pid": os.getpid(),
                "enqueued": self._ticket_enqueued,
                "heartbeat": time.time() if heartbeat is None else heartbeat,
            },
            sort_keys=True,
        ).encode("utf-8")
        try:
            stream.seek(0)
            stream.write(body.ljust(TICKET_BODY_BYTES))
            stream.flush()
        except OSError:
            pass

    def _first_waiter_ahead(self) -> Optional[int]:
        """The ticket of the first live waiter ahead of this one, or None when it is this
        waiter's turn to try the lock (always None without a ticket).

        Dead tickets ahead are removed on the way; stale ones are skipped but left in place.
        """
        if self._ticket is None:
            return None
        now = time.time()
        for number, path in _list_tickets(self.queue_dir):
            if number >= self._ticket:
                break
            state = _probe_ticket(path)
            if state == "dead":
                _remove_quietly(path)
                continue
            if state == "gone":
                continue
            heartbeat = _read_ticket(path).get("heartbeat")
            if isinstance(heartbeat, (int, float)) and now - heartbeat > QUEUE_STALE_S:
                continue
            return number
        return None

    def _dequeue(self) -> None:
        """Leave the queue: unlock, close and remove this waiter's ticket, if it has one."""
        stream, path = self._ticket_stream, self._ticket_path
        self._ticket_stream = None
        self._ticket_path = None
        if stream is None or path is None:
            return
        try:
            _queue_unlock(stream)
        except OSError:
            pass
        finally:
            stream.close()
        # Windows refuses to remove a file while another waiter is probing it, for a moment.
        # If it never succeeds, the unlocked ticket reads as dead and the next waiter removes it.
        for _ in range(50):
            try:
                os.unlink(path)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.002)
            except OSError:
                return

    def _write_metadata(self, status: str = "held") -> None:
        with self._metadata_lock:
            self._metadata.update(
                status=status, owner=self.owner, run_id=self.run_id, pid=os.getpid(),
                process_start_identity=self.process_start_identity, heartbeat=time.time(),
                timeout=self.timeout, lock_path=str(self.lock_path),
            )
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self.metadata_path.name}.", suffix=".tmp", dir=self.metadata_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(self._metadata, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                # On Windows a reader that has the metadata file open (`status`, for example)
                # makes the swap fail for a moment; retry briefly instead of aborting the holder.
                for attempt in range(50):
                    try:
                        os.replace(temporary, self.metadata_path)
                        break
                    except PermissionError:
                        if attempt == 49:
                            raise
                        time.sleep(0.01)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def acquire(self) -> "BenchmarkLease":
        """Wait for and acquire the lease, raising ``TimeoutError`` on expiry.

        First come, first served: this waiter tries the lock only once every waiter that came
        before it has got the lease, given up or died (see the module docstring).
        """
        if self._held:
            return self
        self._stop.clear()
        self._heartbeat_error = None
        self.run_id = secrets.token_hex(16)
        self.process_start_identity = _process_start_identity()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.lock_path, "a+b")
        started = time.monotonic()
        deadline = started + self.timeout
        acquired = False
        try:
            ahead_at_entry = self._enqueue()
            while True:
                first_in_line = self._first_waiter_ahead() is None
                if first_in_line and self._try_lock():
                    acquired = True
                    break
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(f"benchmark lease is held: {self.lock_path}")
                self._refresh_ticket()
                pause = HEAD_POLL_S if first_in_line else QUEUE_POLL_S
                time.sleep(min(pause, max(0.001, deadline - now)))
        finally:
            # Got the lease, gave up or was interrupted: either way this waiter leaves the queue.
            self._dequeue()
            if not acquired:
                self._file.close()
                self._file = None
        self._held = True
        with self._metadata_lock:
            self._metadata.update(
                acquired_at=time.time(),
                queue=self._queue_note,
                queue_ticket=self._ticket,
                queue_ahead_at_entry=ahead_at_entry,
                queue_waited_s=round(time.monotonic() - started, 3),
            )
        try:
            self._write_metadata()
        except BaseException:
            if os.name == "nt":
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
            self._held = False
            raise
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        try:
            self._heartbeat_thread.start()
        except BaseException:
            self._stop.set()
            self._unlock_and_close()
            raise
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            with self._metadata_lock:
                held = self._held
            if held:
                try:
                    self._write_metadata()
                except BaseException as exc:
                    with self._metadata_lock:
                        self._heartbeat_error = exc
                    self._stop.set()
                    return

    def check_health(self) -> None:
        """Raise if background heartbeat publication failed."""
        with self._metadata_lock:
            error = self._heartbeat_error
        if error is not None:
            raise RuntimeError("benchmark lease heartbeat failed") from error

    def heartbeat(self) -> None:
        """Publish a fresh heartbeat while holding the lease."""
        with self._metadata_lock:
            if not self._held:
                raise RuntimeError("benchmark lease is not held")
            self.check_health()
            self._write_metadata()

    def _unlock_and_close(self) -> None:
        with self._metadata_lock:
            self._held = False
        assert self._file is not None
        try:
            if os.name == "nt":
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def release(self) -> None:
        """Release this lease; an unheld or foreign lease cannot unlock it."""
        if not self._held:
            return
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
        # Write the terminal record while our OS lock is still held.  A new
        # owner can then replace it atomically after acquiring the same file.
        with self._metadata_lock:
            try:
                self._write_metadata(status="released")
            finally:
                self._unlock_and_close()

    def __enter__(self) -> "BenchmarkLease":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def queue_status(path: Optional[os.PathLike[str] | str] = None) -> dict[str, Any]:
    """The holder's last published metadata and the queue, lowest ticket first. Changes nothing.

    A ticket's ``state`` is ``live`` (its waiter is waiting), ``stale`` (alive, but it has not
    refreshed for ``QUEUE_STALE_S`` seconds, so the others skip it), ``dead`` (its process ended;
    the next waiter removes it) or ``gone``.
    """
    lock_path = (Path(path) if path is not None else _default_lock_path()).resolve()
    try:
        holder = json.loads(Path(str(lock_path) + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        holder = {}
    now = time.time()
    queue = []
    for number, ticket_path in _list_tickets(Path(str(lock_path) + QUEUE_SUFFIX)):
        body = _read_ticket(ticket_path)
        state = _probe_ticket(ticket_path)
        heartbeat = body.get("heartbeat")
        age = now - heartbeat if isinstance(heartbeat, (int, float)) else None
        if state == "live" and age is not None and age > QUEUE_STALE_S:
            state = "stale"
        enqueued = body.get("enqueued")
        queue.append(
            {
                "ticket": number,
                "state": state,
                "owner": body.get("owner"),
                "pid": body.get("pid"),
                "waiting_s": now - enqueued if isinstance(enqueued, (int, float)) else None,
                "heartbeat_age_s": age,
            }
        )
    return {"lock_path": str(lock_path), "holder": holder, "queue": queue}


def _describe(report: dict[str, Any]) -> str:
    now = time.time()
    holder = report["holder"]
    lines = [f"lease {report['lock_path']}"]
    if holder.get("status") == "held":
        text = f"held by {holder.get('owner')} (pid {holder.get('pid')})"
        since = holder.get("acquired_at")
        if isinstance(since, (int, float)):
            text += f" for {(now - since) / 60:.1f} min"
        beat = holder.get("heartbeat")
        if isinstance(beat, (int, float)):
            text += f", heartbeat {now - beat:.0f} s ago"
            if now - beat > 60:
                text += " (old: the holder may have exited without releasing)"
        lines.append(text)
    elif holder:
        lines.append(f"free (last holder {holder.get('owner')}, {holder.get('status')})")
    else:
        lines.append("free (no holder has published metadata)")
    if not report["queue"]:
        lines.append("queue: empty")
    for position, ticket in enumerate(report["queue"], 1):
        waiting = ticket["waiting_s"]
        waited = f", waiting {waiting / 60:.1f} min" if waiting is not None else ""
        lines.append(
            f"  {position}. ticket {ticket['ticket']} [{ticket['state']}] {ticket['owner']} "
            f"(pid {ticket['pid']}){waited}"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show who holds the benchmark lease and who waits.")
    parser.add_argument("command", choices=["status"])
    parser.add_argument("--path", help="the lease file (default: the machine-wide one)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    report = queue_status(args.path)
    print(json.dumps(report, indent=2) if args.json else _describe(report))
    return 0


__all__ = ["BenchmarkLease", "queue_status"]


if __name__ == "__main__":
    raise SystemExit(main())
