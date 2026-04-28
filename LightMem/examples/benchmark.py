import os
import json
import time
import csv
import copy
import statistics
from typing import Dict, List, Any, Optional

from lightmem.memory.lightmem import LightMemory


# =========================
# CONFIG
# =========================

LLMLINGUA_MODEL_PATH = os.getenv(
    "LLMLINGUA_MODEL_PATH",
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
)
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DATA_PATH = os.getenv(
    "DATA_PATH",
    "/Users/ambikasharan/Desktop/lightmem/LightMem/experiments/longmemeval/longmemeval_s.json",
)
QDRANT_DATA_DIR = os.getenv(
    "QDRANT_DATA_DIR",
    "/Users/ambikasharan/Desktop/lightmem/LightMem/qdrant_data",
)

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "phi")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((p / 100.0) * (len(values_sorted) - 1)))
    idx = max(0, min(idx, len(values_sorted) - 1))
    return values_sorted[idx]


def clone_turn_messages(turn_messages: List[Dict[str, Any]], timestamp: str) -> List[Dict[str, Any]]:
    copied = copy.deepcopy(turn_messages)
    for msg in copied:
        msg["time_stamp"] = timestamp
    return copied


def config_label(max_items: Optional[int], max_sessions_per_item: Optional[int]) -> str:
    items_str = "allitems" if max_items is None else f"{max_items}item"
    sessions_str = "allsessions" if max_sessions_per_item is None else f"{max_sessions_per_item}session"
    return f"{items_str}_{sessions_str}"


# =========================
# LIGHTMEM CONFIG
# =========================

def build_config(collection_name: str, streaming_mode: bool) -> Dict[str, Any]:
    """
    Latency-focused benchmark config:
    - baseline: compress at write time
    - streaming: precompute compression via ingest_message_stream
    - metadata/text summary disabled so no external LLM work
    - topic segmentation disabled to avoid unrelated segmenter issues
    """
    return {
        "pre_compress": True,
        "pre_compress_streaming": streaming_mode,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cpu",
                    "use_llmlingua2": True,
            },
            "compress_config": {
                "rate": 0.6,
                },
                "compress_config": {
                    "rate": 0.6,
                },
            },
        },

        # Disabled for pure compression benchmark
        "topic_segment": False,
        "precomp_topic_shared": False,

        "messages_use": "user_only",
        "metadata_generate": False,
        "text_summary": False,

        "memory_manager": {
            "model_name": "ollama",
            "configs": {
                "model": OLLAMA_MODEL_NAME,
                "host": OLLAMA_HOST,
                "max_tokens": 512,
            },
        },

        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cpu"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f"{QDRANT_DATA_DIR}/{collection_name}",
            },
        },
        "update": "offline",
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "log_dir": "logs",
            "log_filename_prefix": f"benchmark_{'stream' if streaming_mode else 'base'}",
            "console_enabled": True,
            "file_level": "DEBUG",
        },
    }


def load_lightmem(collection_name: str, streaming_mode: bool) -> LightMemory:
    config = build_config(collection_name, streaming_mode)
    return LightMemory.from_config(config)


# =========================
# BENCHMARK CORE
# =========================

def run_mode(
    data: List[Dict[str, Any]],
    streaming_mode: bool,
    max_items: Optional[int] = 1,
    max_sessions_per_item: Optional[int] = 1,
) -> Dict[str, Any]:
    mode_name = "streaming" if streaming_mode else "baseline"
    collection_name = f"bench_{mode_name}_{int(time.time() * 1000)}"
    lightmem = load_lightmem(collection_name=collection_name, streaming_mode=streaming_mode)

    precompute_latencies: List[float] = []
    add_memory_latencies: List[float] = []
    total_turns = 0
    total_sessions_used = 0

    run_start = time.perf_counter()

    selected_items = data[:max_items] if max_items is not None else data

    for item in selected_items:
        all_sessions = item.get("haystack_sessions", [])
        all_timestamps = item.get("haystack_dates", [])

        if max_sessions_per_item is not None:
            sessions = all_sessions[:max_sessions_per_item]
            timestamps = all_timestamps[:max_sessions_per_item]
        else:
            sessions = all_sessions
            timestamps = all_timestamps

        total_sessions_used += len(sessions)

        for session_idx, (session, timestamp) in enumerate(zip(sessions, timestamps)):
            session_copy = copy.deepcopy(session)

            while session_copy and session_copy[0]["role"] != "user":
                session_copy.pop(0)

            num_turns = len(session_copy) // 2

            for turn_idx in range(num_turns):
                turn_messages = session_copy[turn_idx * 2 : turn_idx * 2 + 2]

                if len(turn_messages) < 2:
                    continue
                if turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                    continue

                turn_messages = clone_turn_messages(turn_messages, timestamp)
                is_last_turn = (
                    session_idx == len(sessions) - 1 and turn_idx == num_turns - 1
                )

                if streaming_mode:
                    t_pre = time.perf_counter()
                    lightmem.ingest_message_stream(turn_messages)
                    precompute_latencies.append(time.perf_counter() - t_pre)

                t_add = time.perf_counter()
                lightmem.add_memory(
                    messages=turn_messages,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn,
                )
                add_memory_latencies.append(time.perf_counter() - t_add)
                total_turns += 1

    total_runtime = time.perf_counter() - run_start
    stream_stats = getattr(lightmem, "streaming_precompress_stats", {})
    token_stats = lightmem.get_token_statistics()

    return {
        "mode": mode_name,
        "max_items": max_items,
        "max_sessions_per_item": max_sessions_per_item,
        "num_items": len(selected_items),
        "num_sessions": total_sessions_used,
        "num_turns": total_turns,
        "total_runtime_sec": round(total_runtime, 6),

        "avg_add_memory_latency_sec": round(safe_mean(add_memory_latencies), 6),
        "p50_add_memory_latency_sec": round(safe_percentile(add_memory_latencies, 50), 6),
        "p95_add_memory_latency_sec": round(safe_percentile(add_memory_latencies, 95), 6),
        "p99_add_memory_latency_sec": round(safe_percentile(add_memory_latencies, 99), 6),

        "avg_precompute_latency_sec": round(safe_mean(precompute_latencies), 6),
        "p50_precompute_latency_sec": round(safe_percentile(precompute_latencies, 50), 6),
        "p95_precompute_latency_sec": round(safe_percentile(precompute_latencies, 95), 6),
        "p99_precompute_latency_sec": round(safe_percentile(precompute_latencies, 99), 6),

        "cache_hits": stream_stats.get("cache_hits", 0),
        "cache_misses": stream_stats.get("cache_misses", 0),
        "precompute_calls": stream_stats.get("precompute_calls", 0),
        "total_precompute_time_sec": round(stream_stats.get("precompute_time", 0.0), 6),

        "total_llm_calls": token_stats.get("summary", {}).get("total_llm_calls", 0),
        "total_llm_tokens": token_stats.get("summary", {}).get("total_llm_tokens", 0),
        "total_embedding_calls": token_stats.get("summary", {}).get("total_embedding_calls", 0),
        "total_embedding_tokens": token_stats.get("summary", {}).get("total_embedding_tokens", 0),

        "stage_compress_time_sec": round(lightmem.token_stats.get("stage_compress_time", 0.0), 6),
        "stage_segment_time_sec": round(lightmem.token_stats.get("stage_segment_time", 0.0), 6),
        "stage_llm_extract_time_sec": round(lightmem.token_stats.get("stage_llm_extract_time", 0.0), 6),
        "stage_db_insert_time_sec": round(lightmem.token_stats.get("stage_db_insert_time", 0.0), 6),
    }


def run_experiment_pair(
    data: List[Dict[str, Any]],
    max_items: Optional[int],
    max_sessions_per_item: Optional[int],
) -> List[Dict[str, Any]]:
    label = config_label(max_items, max_sessions_per_item)
    print(f"\nRunning config: {label}")

    baseline = run_mode(
        data=data,
        streaming_mode=False,
        max_items=max_items,
        max_sessions_per_item=max_sessions_per_item,
    )
    baseline["experiment"] = label

    streaming = run_mode(
        data=data,
        streaming_mode=True,
        max_items=max_items,
        max_sessions_per_item=max_sessions_per_item,
    )
    streaming["experiment"] = label

    return [baseline, streaming]


# =========================
# OUTPUT
# =========================

def save_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results_json(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# =========================
# MAIN
# =========================

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")

    data = load_dataset(DATA_PATH)

    # Try several settings.
    # Adjust this grid however you want.
    experiment_grid = [
        {"max_items": 1, "max_sessions_per_item": 1},
        {"max_items": 1, "max_sessions_per_item": 3},
        {"max_items": 1, "max_sessions_per_item": 5},
        {"max_items": 1, "max_sessions_per_item": 9},
        {"max_items": 2, "max_sessions_per_item": 1},
        {"max_items": 2, "max_sessions_per_item": 3},
        {"max_items": 3, "max_sessions_per_item": 3},
        {"max_items": 100, "max_sessions_per_item": 1},
        {"max_items": 273, "max_sessions_per_item": 1},

    ]

    all_rows: List[Dict[str, Any]] = []

    overall_start = time.perf_counter()

    for exp in experiment_grid:
        rows = run_experiment_pair(
            data=data,
            max_items=exp["max_items"],
            max_sessions_per_item=exp["max_sessions_per_item"],
        )
        all_rows.extend(rows)

    overall_runtime = time.perf_counter() - overall_start

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        RESULTS_DIR,
        f"benchmark_streaming_precompress_grid_{timestamp}.csv",
    )
    json_path = os.path.join(
        RESULTS_DIR,
        f"benchmark_streaming_precompress_grid_{timestamp}.json",
    )

    save_results_csv(all_rows, csv_path)
    save_results_json(all_rows, json_path)

    print("\n=== Benchmark Results ===")
    for row in all_rows:
        print(json.dumps(row, indent=2))

    print(f"\nCompleted all runs in {overall_runtime:.2f} sec")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()