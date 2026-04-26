from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from benchmark.benchmark_utils import atomic_write_json
from powerllm.models.model_resolver import get_model_spec


def _require_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY in your environment.")
    return api_key


def _build_rewrite_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "bm25_query": {"type": "string"},
        },
        "required": ["bm25_query"],
    }


def _build_rewrite_generation_config(
    *,
    timeout_seconds: int | None = None,
) -> types.GenerateContentConfigDict:
    config: types.GenerateContentConfigDict = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_json_schema": _build_rewrite_response_schema(),
    }
    if timeout_seconds is not None:
        config["http_options"] = {"timeout": timeout_seconds * 1000}
    return config


def _resolve_gemini_model_name(chat_model_id: str) -> str:
    spec = get_model_spec(chat_model_id)
    provider = spec.get("provider")
    if provider != "google_genai":
        raise ValueError(
            f"Gemini Batch API requires a google_genai model, got provider={provider!r}."
        )
    model_name = spec["model_name"]
    provider_prefix = "google_genai:"
    if model_name.startswith(provider_prefix):
        return model_name[len(provider_prefix) :]
    return model_name


def _build_rewrite_prompt(query: str, bm25_k: int) -> str:
    return (
        "You rewrite user questions into a BM25 retrieval query.\n"
        "Return ONLY JSON object: {\"bm25_query\": \"...\"}.\n"
        f"Use at most {min(max(bm25_k, 1), 10)} concise keyword phrases.\n"
        "Keep important names, dates, section labels, and identifiers if present.\n"
        "Do not invent facts.\n\n"
        f"Question: {query}"
    )


def _parse_bm25_query(response_text: str) -> str:
    payload = json.loads(response_text)
    if not isinstance(payload, dict):
        raise ValueError("Batch response is not a JSON object.")
    bm25_query = payload.get("bm25_query")
    if not isinstance(bm25_query, str) or not bm25_query.strip():
        raise ValueError("Batch response missing non-empty 'bm25_query'.")
    return bm25_query.strip()


def _build_inlined_rewrite_request(
    *,
    cache_key: str,
    question: str,
    bm25_k: int,
) -> types.InlinedRequestDict:
    return {
        "contents": _build_rewrite_prompt(question, bm25_k),
        "metadata": {"cache_key": cache_key},
        "config": _build_rewrite_generation_config(),
    }


def _to_cache_record(
    *,
    bm25_query: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "bm25_query": bm25_query,
        "bm25_query_rewrite_applied": error is None,
        "bm25_query_rewrite_error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_missing_items(
    missing_items: list[dict[str, str]],
) -> list[dict[str, str]]:
    unique_by_key: dict[str, dict[str, str]] = {}
    for item in missing_items:
        cache_key = item.get("cache_key")
        question = item.get("question")
        if not isinstance(cache_key, str) or not cache_key:
            continue
        if not isinstance(question, str) or not question:
            continue
        unique_by_key.setdefault(cache_key, {"cache_key": cache_key, "question": question})
    return list(unique_by_key.values())


def _poll_batch_job(
    *,
    client: genai.Client,
    job_name: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
):
    """
    Poll Gemini batch job until completion or timeout. Returns the completed job object.
     - Raises TimeoutError if job does not complete within timeout_seconds.
     - Raises RuntimeError if job completes with an error.
    """
    started_at = time.time()
    deadline = started_at + timeout_seconds
    last_state = None
    while True:
        job = client.batches.get(name=job_name)
        state = str(getattr(job, "state", None))
        elapsed_seconds = int(time.time() - started_at)
        if state != last_state:
            print(
                f"Gemini batch job {job_name} state: {state} "
                f"(elapsed={elapsed_seconds}s)"
            )
            last_state = state
        else:
            print(
                f"Gemini batch job {job_name} heartbeat: still {state} "
                f"(elapsed={elapsed_seconds}s)"
            )

        if job.done:
            return job
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for Gemini batch job: {job_name}")
        time.sleep(poll_interval_seconds)



def populate_bm25_cache_via_gemini_batch(
    *,
    missing_items: list[dict[str, str]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    chat_model_id: str,
    bm25_k: int,
    batch_size: int = 1000, # The max size to send requests in batches.
    poll_interval_seconds: int = 5,
    timeout_seconds: int = 1800, # 30 minutes
) -> dict[str, dict[str, Any]]:
    """
    Populate missing BM25 cache entries using Gemini Batch API (google-genai SDK).

    missing_items item format:
      {"cache_key": str, "question": str}
    """
    if not missing_items:
        return cache

    api_key = _require_google_api_key()

    model_name = _resolve_gemini_model_name(chat_model_id)
    client = genai.Client(api_key=api_key)

    tasks = _normalize_missing_items(missing_items)
    total = len(tasks)
    print(f"Submitting {total} BM25 rewrites via Gemini Batch API...")

    for start in range(0, total, batch_size):
        chunk = tasks[start : start + batch_size]
        inlined_requests = [
            _build_inlined_rewrite_request(
                cache_key=item["cache_key"],
                question=item["question"],
                bm25_k=bm25_k,
            )
            for item in chunk
        ]

        batch_job = client.batches.create(
            model=model_name,
            src=inlined_requests,
            config=types.CreateBatchJobConfig(
                display_name=f"bm25-rewrite-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            ),
        )
        print(f"Created Gemini batch job: {batch_job.name}")
        completed_job = _poll_batch_job(
            client=client,
            job_name=batch_job.name or "",
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )

        responses = getattr(getattr(completed_job, "dest", None), "inlined_responses", None)
        if not isinstance(responses, list):
            raise RuntimeError("Gemini batch job completed without inlined responses.")

        if len(responses) != len(chunk):
            raise RuntimeError(
                f"Gemini batch response size mismatch: expected {len(chunk)}, got {len(responses)}"
            )

        for index, response_item in enumerate(responses):
            metadata = getattr(response_item, "metadata", None) or {}
            cache_key = metadata.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                cache_key = chunk[index]["cache_key"]

            item_error = getattr(response_item, "error", None)
            if item_error is not None:
                message = getattr(item_error, "message", None) or str(item_error)
                raise RuntimeError(f"Gemini batch rewrite failed for {cache_key}: {message}")

            model_response = getattr(response_item, "response", None)
            if model_response is None:
                raise RuntimeError(f"Missing model response for {cache_key}")
            response_text = getattr(model_response, "text", None)
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError(f"Empty model response for {cache_key}")

            bm25_query = _parse_bm25_query(response_text)
            cache[cache_key] = _to_cache_record(bm25_query=bm25_query, error=None)

        atomic_write_json(cache_path, cache)
        print(
            f"Gemini batch progress: {min(start + len(chunk), total)}/{total} cached."
        )

    return cache


def _generate_bm25_query_once(
    *,
    api_key: str,
    model_name: str,
    question: str,
    bm25_k: int,
    timeout_seconds: int,
) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=_build_rewrite_prompt(question, bm25_k),
        config=_build_rewrite_generation_config(timeout_seconds=timeout_seconds),
    )
    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Empty response text from Gemini generate_content.")
    return _parse_bm25_query(response_text)


def _rewrite_with_retry(
    *,
    api_key: str,
    model_name: str,
    cache_key: str,
    question: str,
    bm25_k: int,
    timeout_seconds: int,
    max_attempts: int,
    backoff_seconds: int,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            bm25_query = _generate_bm25_query_once(
                api_key=api_key,
                model_name=model_name,
                question=question,
                bm25_k=bm25_k,
                timeout_seconds=timeout_seconds,
            )
            return {
                "cache_key": cache_key,
                "question": question,
                "bm25_query": bm25_query,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_attempts:
                break
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[concurrent] retry cache_key={cache_key} attempt={attempt + 1}/{max_attempts} "
                f"after {sleep_seconds}s (error={last_error})"
            )
            time.sleep(sleep_seconds)

    return {
        "cache_key": cache_key,
        "question": question,
        "bm25_query": question,
        "error": last_error or "Unknown rewrite error",
    }


def populate_bm25_cache_via_concurrent(
    *,
    missing_items: list[dict[str, str]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    chat_model_id: str,
    bm25_k: int,
    thread_limit: int = 10,
    cool_after_n: int = 100,
    wait_cooling_time: int = 10,
    timeout_seconds: int = 120,
    max_attempts: int = 3,
    backoff_seconds: int = 2,
) -> dict[str, dict[str, Any]]:
    """
    Populate missing BM25 cache entries using concurrent Gemini generate_content calls.

    Each query is handled by a single worker task with in-task retries to avoid duplicate records.
    """
    if not missing_items:
        return cache

    api_key = _require_google_api_key()

    if thread_limit < 1:
        raise ValueError("thread_limit must be >= 1")
    if cool_after_n < 1:
        raise ValueError("cool_after_n must be >= 1")
    if wait_cooling_time < 0:
        raise ValueError("wait_cooling_time must be >= 0")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be >= 0")

    model_name = _resolve_gemini_model_name(chat_model_id)

    tasks = _normalize_missing_items(missing_items)
    total = len(tasks)
    print(
        f"Submitting {total} BM25 rewrites via Gemini concurrent API "
        f"(thread_limit={thread_limit}, cool_after_n={cool_after_n}, wait={wait_cooling_time}s)"
    )

    success_count = 0
    failure_count = 0
    processed = 0

    for start in range(0, total, cool_after_n):
        window = tasks[start : start + cool_after_n]
        window_index = start // cool_after_n + 1
        window_total = (total + cool_after_n - 1) // cool_after_n
        print(
            f"[concurrent] window {window_index}/{window_total}: "
            f"processing {len(window)} queries"
        )

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=thread_limit) as executor:
            futures = [
                executor.submit(
                    _rewrite_with_retry,
                    api_key=api_key,
                    model_name=model_name,
                    cache_key=item["cache_key"],
                    question=item["question"],
                    bm25_k=bm25_k,
                    timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts,
                    backoff_seconds=backoff_seconds,
                )
                for item in window
            ]
            for future in as_completed(futures):
                results.append(future.result())

        for result in results:
            cache_key = result["cache_key"]
            question = result["question"]
            error = result.get("error")
            if error is None:
                success_count += 1
            else:
                failure_count += 1
            cache[cache_key] = _to_cache_record(
                bm25_query=result["bm25_query"],
                error=error,
            )
            # Keep deterministic fallback explicit for failed rewrites.
            if error is not None and cache[cache_key]["bm25_query"] != question:
                cache[cache_key]["bm25_query"] = question

        processed += len(window)
        atomic_write_json(cache_path, cache)
        print(
            f"[concurrent] window {window_index}/{window_total} done: "
            f"processed={processed}/{total}, success={success_count}, failed={failure_count}"
        )

        if start + cool_after_n < total and wait_cooling_time > 0:
            print(
                f"[concurrent] cooling down for {wait_cooling_time}s "
                "to reduce RPM pressure..."
            )
            time.sleep(wait_cooling_time)

    print(
        f"[concurrent] completed: total={total}, success={success_count}, failed={failure_count}"
    )
    return cache
