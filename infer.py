"""Run a vision model through an OpenAI-compatible Chat Completions endpoint."""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import common as ec


def parse_prediction(raw, track):
    """Parse the decision fields and retain incomplete answers for evaluation."""
    degraded = False
    try:
        answer = ec.extract_json(raw)
    except (ValueError, TypeError):
        answer = ec.salvage_json(raw)
        degraded = True
    result = {"pred_mode": ec.norm_mode(answer.get("mode"), track)}
    for field in ("risk_source", "reasoning", "safe_next_step",
                  "clarifying_question", "safety_constraints"):
        result[f"pred_{field}"] = ec.norm_text(answer.get(field))
    if track == "trajectory":
        result["pred_progress"] = ec.norm_enum(answer.get("progress"), ec.PROGRESS_VALUES)
        result["pred_recoverability"] = ec.norm_enum(
            answer.get("recoverability"), ec.RECOVERABILITY_VALUES)
    if degraded:
        result["parse_degraded"] = True
    return result


def run_track(client, args, track):
    rows = ec.load_subset(track)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {ec.subset_path(track)}")
    rows = rows[:args.limit] if args.limit else rows
    path = Path(ec.pred_path(args.tag, track))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Plain JSON metadata prevents accidental reuse after changing a model/prompt.
    config_path = path.with_suffix(".config.json")
    data_stat = Path(ec.subset_path(track)).stat()
    config = {"data_dir": str(ec.DATA_DIR), "data_size": data_stat.st_size,
              "data_mtime_ns": data_stat.st_mtime_ns,
              "model": args.model, "base_url": args.base_url,
              "max_tokens": args.max_tokens, "jpeg": args.jpeg,
              "prompt": ec.load_prompt(f"judge_{track}_v2")}
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration changed; use a new --tag")
    if path.exists() and not config_path.exists():
        raise ValueError("Existing predictions have no run configuration; use a new --tag")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    previous = ec.read_jsonl(path, repair=True) if path.exists() else []
    ec.check_subset_match(previous, str(path))
    done = {r["id"] for r in previous if not r.get("request_error")}
    completed = sum(r["id"] in done for r in rows)
    failed = 0
    with path.open("a", encoding="utf-8") as stream, \
            tqdm(total=len(rows), initial=completed, desc=track, dynamic_ncols=True) as bar:
        for row in rows:
            if row["id"] in done:
                continue
            record = {"id": row["id"], "track": track, "model": args.tag,
                      "subset": ec.subset_name(), "pred_mode": None}
            # Only question, context and images go to the model. No GT/preference.
            try:
                content = [{"type": "text", "text": ec.judge_prompt(row, track)}]
                encoder = ec.encode_image_jpeg if args.jpeg else ec.encode_image
                content.extend({"type": "image_url", "image_url": {"url": encoder(p)}}
                               for p in ec.images_of(row, track))
                options = {"max_tokens": args.max_tokens} if args.max_tokens else {}
                response = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": content}],
                    **options)
                record["raw"] = response.choices[0].message.content or ""
                record["finish_reason"] = response.choices[0].finish_reason
            except Exception as error:
                record["request_error"] = str(error)
                failed += 1
                tqdm.write(f"[request failed] {row['id']}: {error}")
            else:
                try:
                    record.update(parse_prediction(record["raw"], track))
                except (ValueError, TypeError, AttributeError) as error:
                    record["parse_error"] = str(error)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            bar.update(1)
            bar.set_postfix(request_failed=failed)
    # Keep the last attempt for each ID, in dataset order. Replace atomically.
    records = {r["id"]: r for r in ec.read_jsonl(path)}
    ordered = [records[r["id"]] for r in ec.load_subset(track) if r["id"] in records]
    ec.write_jsonl(str(path), ordered)
    print(f"Predictions: {path}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name served by your endpoint")
    parser.add_argument("--tag", required=True, help="Short output name, e.g. my-model")
    parser.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--limit", type=int, default=0, help="First N samples per track; 0 = all")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Optional output budget, if supported by your endpoint")
    parser.add_argument("--jpeg", action="store_true",
                        help="Encode images as JPEG at original dimensions for smaller requests")
    ec.add_common_args(parser)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("--tag must contain only letters, numbers, dots, underscores or hyphens")
    if args.limit < 0 or (args.max_tokens is not None and args.max_tokens <= 0):
        parser.error("--limit must be >= 0 and --max-tokens must be > 0")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        parser.error("Set OPENAI_API_KEY (use EMPTY for a local endpoint without authentication)")
    ec.set_data_dir(args.data_dir)
    with OpenAI(base_url=args.base_url, api_key=key, timeout=180, max_retries=2) as client:
        failures = sum(run_track(client, args, track) for track in ec.resolve_tracks(args.track))
    if failures:
        raise SystemExit("Some requests failed. Run the same command to retry them.")


if __name__ == "__main__":
    main()
