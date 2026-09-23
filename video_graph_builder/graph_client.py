import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    STORAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMP_RESULTS_DIR = os.path.join(STORAGE_PATH, "temp_results")
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

CAPTION_REQUEST_RETRIES = int(os.getenv("CAPTION_REQUEST_RETRIES", "12"))
CAPTION_REQUEST_RETRY_SLEEP_SEC = float(os.getenv("CAPTION_REQUEST_RETRY_SLEEP_SEC", "5"))
CAPTION_SERVER_BASE_PORT = int(os.getenv("RISE_CAPTION_SERVER_BASE_PORT", "6000"))


def generate_temp_filename(prefix="caption", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return os.path.join(TEMP_RESULTS_DIR, f"{prefix}_{timestamp}_{rand_part}{suffix}")


def split_list(items, n):
    n = max(1, n)
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def fetch(port, path, endpoint="caption_entities"):
    url = f"http://127.0.0.1:{port}/{endpoint}"
    last_exc = None
    for attempt in range(1, CAPTION_REQUEST_RETRIES + 1):
        try:
            response = requests.get(url, params={"name": path}, timeout=1800)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= CAPTION_REQUEST_RETRIES:
                break
            print(
                f"[graph-client] retrying endpoint={endpoint} port={port} "
                f"attempt={attempt}/{CAPTION_REQUEST_RETRIES} after error: {exc}"
            )
            time.sleep(CAPTION_REQUEST_RETRY_SLEEP_SEC)
    raise last_exc


def caption_entities_batch(items, num_servers, base_port=None, endpoint="caption_entities"):
    """items: [{"segment_id": str, "image": base64-png-str}, ...]. Shards the
    request across `num_servers` /caption_entities vLLM servers (ports
    base_port..base_port+n-1), following the same write-task-file -> GET
    ?name= -> poll/read/delete _results.json convention as
    train_examples/reward_function/cot_val.py's generate_results. Returns a
    flat list of {"segment_id","caption","entities"} dicts (order not
    guaranteed to match `items`; callers should key by segment_id).
    """
    if not items:
        return []

    base_port = base_port if base_port is not None else CAPTION_SERVER_BASE_PORT
    n = max(1, min(num_servers, len(items)))
    shards = split_list(items, n)
    paths = [generate_temp_filename(prefix=f"caption_{i}") for i in range(n)]

    for path, shard in zip(paths, shards):
        with open(path, "w") as f:
            json.dump(shard, f)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [
            executor.submit(fetch, base_port + i, paths[i], endpoint)
            for i in range(n)
        ]
        for future in as_completed(futures):
            future.result()

    results = []
    for path in paths:
        result_path = path.replace(".json", "_results.json")
        with open(result_path, "r") as f:
            results.extend(json.load(f))
        os.remove(result_path)

    return results
