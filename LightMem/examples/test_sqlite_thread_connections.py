#!/usr/bin/env python3
"""Compare shared-connection SQLite writes vs per-thread SQLiteManager writes."""

import os
import sqlite3
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import importlib.util

# Allow running this file directly from LightMem/examples.
ROOT = Path(__file__).resolve().parents[1]
STORAGE_FILE = ROOT / "src" / "lightmem" / "memory_toolkits" / "memories" / "layers" / "baselines" / "mem0" / "memory" / "storage.py"

_spec = importlib.util.spec_from_file_location("mem0_storage", STORAGE_FILE)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load storage module at {STORAGE_FILE}")
_mem0_storage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mem0_storage)
SQLiteManager = _mem0_storage.SQLiteManager


def run_shared_connection_test(db_path: str, threads: int = 8, writes_per_thread: int = 50) -> Dict[str, Any]:
    """Intentionally share one sqlite connection across threads to reproduce thread errors."""
    conn = sqlite3.connect(db_path)  # check_same_thread=True by default
    conn.execute("DROP TABLE IF EXISTS shared_test")
    conn.execute("CREATE TABLE shared_test (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id INTEGER, payload TEXT)")
    conn.commit()

    lock = threading.Lock()
    error_messages = []
    success_count = 0

    def worker(tid: int) -> None:
        nonlocal success_count
        for i in range(writes_per_thread):
            try:
                conn.execute("INSERT INTO shared_test (thread_id, payload) VALUES (?, ?)", (tid, f"msg-{i}"))
                with lock:
                    success_count += 1
            except Exception as exc:
                with lock:
                    error_messages.append(f"{type(exc).__name__}: {exc}")
                break

    jobs = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    try:
        conn.commit()
    except Exception:
        pass

    row_count = conn.execute("SELECT COUNT(*) FROM shared_test").fetchone()[0]
    conn.close()

    return {
        "mode": "shared_connection",
        "threads": threads,
        "writes_attempted": threads * writes_per_thread,
        "writes_succeeded": success_count,
        "db_rows": row_count,
        "error_count": len(error_messages),
        "sample_errors": sorted(set(error_messages))[:5],
    }


def run_per_thread_manager_test(db_path: str, threads: int = 8, writes_per_thread: int = 50) -> Dict[str, Any]:
    """Use SQLiteManager where each thread gets its own connection."""
    mgr = SQLiteManager(db_path)
    mgr.reset()

    lock = threading.Lock()
    error_messages = []
    success_count = 0

    now = datetime.now(timezone.utc).isoformat()

    def worker(tid: int) -> None:
        nonlocal success_count
        memory_id = f"memory-{tid}"
        try:
            for i in range(writes_per_thread):
                mgr.add_history(
                    memory_id=memory_id,
                    old_memory=f"old-{i}",
                    new_memory=f"new-{i}",
                    event="UPDATE",
                    created_at=now,
                    updated_at=now,
                    actor_id=f"actor-{tid}",
                    role="user",
                )
                with lock:
                    success_count += 1
        except Exception as exc:
            with lock:
                error_messages.append(f"{type(exc).__name__}: {exc}")
        finally:
            mgr.close_thread_conn()

    jobs = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    # Count rows directly to avoid relying on memory_id-specific ordering assumptions.
    rows = mgr.execute("SELECT COUNT(*) FROM history")
    row_count = rows[0][0] if rows else 0
    mgr.close()

    return {
        "mode": "per_thread_connections",
        "threads": threads,
        "writes_attempted": threads * writes_per_thread,
        "writes_succeeded": success_count,
        "db_rows": row_count,
        "error_count": len(error_messages),
        "sample_errors": sorted(set(error_messages))[:5],
    }


def print_result(result: Dict[str, Any]) -> None:
    print(f"\n=== {result['mode']} ===")
    print(f"threads: {result['threads']}")
    print(f"writes_attempted: {result['writes_attempted']}")
    print(f"writes_succeeded: {result['writes_succeeded']}")
    print(f"db_rows: {result['db_rows']}")
    print(f"error_count: {result['error_count']}")
    if result["sample_errors"]:
        print("sample_errors:")
        for err in result["sample_errors"]:
            print(f"  - {err}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sqlite-thread-test-") as td:
        shared_db = os.path.join(td, "shared.db")
        manager_db = os.path.join(td, "manager.db")

        shared_result = run_shared_connection_test(shared_db)
        manager_result = run_per_thread_manager_test(manager_db)

        print_result(shared_result)
        print_result(manager_result)

        if manager_result["error_count"] == 0 and manager_result["writes_succeeded"] == manager_result["writes_attempted"]:
            print("\nPASS: per-thread manager writes completed without SQLite thread errors.")
        else:
            print("\nFAIL: per-thread manager test still has errors.")


if __name__ == "__main__":
    main()

