#!/usr/bin/env python3
"""Threading regression test for SQLiteManager.

This script runs two scenarios:
1) Intentionally shared sqlite3 connection across many threads (expected failure).
2) SQLiteManager with per-thread connections (expected success).
"""

import importlib.util
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE_PATH = (
    ROOT
    / "src"
    / "lightmem"
    / "memory_toolkits"
    / "memories"
    / "layers"
    / "baselines"
    / "mem0"
    / "memory"
    / "storage.py"
)

spec = importlib.util.spec_from_file_location("mem0_storage", STORAGE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {STORAGE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SQLiteManager = mod.SQLiteManager


def run_shared_connection_case(db_path: str, threads: int, writes_per_thread: int) -> dict:
    conn = sqlite3.connect(db_path)  # check_same_thread=True by default
    conn.execute("DROP TABLE IF EXISTS shared_test")
    conn.execute("CREATE TABLE shared_test (id INTEGER PRIMARY KEY, tid INTEGER, payload TEXT)")
    conn.commit()

    lock = threading.Lock()
    successes = 0
    errors = []

    def worker(tid: int) -> None:
        nonlocal successes
        for i in range(writes_per_thread):
            try:
                conn.execute("INSERT INTO shared_test(tid, payload) VALUES(?, ?)", (tid, f"m-{i}"))
                with lock:
                    successes += 1
            except Exception as exc:  # expected in this case
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                break

    jobs = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    row_count = conn.execute("SELECT COUNT(*) FROM shared_test").fetchone()[0]
    conn.close()
    return {
        "name": "shared_connection",
        "attempted": threads * writes_per_thread,
        "successes": successes,
        "rows": row_count,
        "error_count": len(errors),
        "sample_errors": sorted(set(errors))[:3],
    }


def run_sqlite_manager_case(db_path: str, threads: int, writes_per_thread: int) -> dict:
    manager = SQLiteManager(db_path)
    manager.reset()

    lock = threading.Lock()
    successes = 0
    errors = []
    now = datetime.now(timezone.utc).isoformat()

    def worker(tid: int) -> None:
        nonlocal successes
        try:
            for i in range(writes_per_thread):
                manager.add_history(
                    memory_id=f"m-{tid}",
                    old_memory=f"old-{i}",
                    new_memory=f"new-{i}",
                    event="UPDATE",
                    created_at=now,
                    updated_at=now,
                    actor_id=f"actor-{tid}",
                    role="user",
                )
                with lock:
                    successes += 1
        except Exception as exc:
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            manager.close_thread_conn()

    jobs = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    rows = manager.connection.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    manager.close()
    return {
        "name": "sqlite_manager_per_thread",
        "attempted": threads * writes_per_thread,
        "successes": successes,
        "rows": rows,
        "error_count": len(errors),
        "sample_errors": sorted(set(errors))[:3],
    }


def print_result(result: dict) -> None:
    print(f"\n=== {result['name']} ===")
    print(f"attempted:   {result['attempted']}")
    print(f"successes:   {result['successes']}")
    print(f"db rows:     {result['rows']}")
    print(f"error_count: {result['error_count']}")
    for err in result["sample_errors"]:
        print(f"  - {err}")


def main() -> None:
    threads = 8
    writes = 50

    with tempfile.TemporaryDirectory(prefix="sqlite-thread-test-") as td:
        shared = run_shared_connection_case(os.path.join(td, "shared.db"), threads, writes)
        fixed = run_sqlite_manager_case(os.path.join(td, "fixed.db"), threads, writes)

    print_result(shared)
    print_result(fixed)

    if fixed["error_count"] == 0 and fixed["successes"] == fixed["attempted"]:
        print("\nPASS: per-thread manager writes succeeded without sqlite thread errors.")
    else:
        print("\nFAIL: per-thread manager scenario still has errors.")


if __name__ == "__main__":
    main()

