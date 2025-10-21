import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def _resolve_path(repo_root: Path, p: str) -> Optional[str]:
    """Resolve image path relative to repo root; return None if missing."""
    if not p:
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = repo_root / p
    try:
        pp = pp.resolve()
    except Exception:
        pass
    return str(pp) if pp.exists() else None


def _gather_tasks_for_file(repo_root: Path, json_path: Path) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
    """
    Inspect a result JSON and gather image paths that need HPS updates.

    Returns
    - image_paths: list of image file paths to score (deduplicated, existing only)
    - mapping: maps a unique key to (json_key, field) so we can write results back
      Keys:
        - For per-turn entries in multipass: f"turn:{turn}"
        - For singlepass top-level: "singlepass"
        - For base image: "base"
      Field is one of: 'human_preference_score' or 'base_image_human_preference_score'
    """
    image_paths: List[str] = []
    mapping: Dict[str, Tuple[str, str]] = {}

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON {json_path}: {e}")
        return image_paths, mapping

    # Base image HPS
    base_hps_missing = False
    base_img_path = None
    if isinstance(data, dict) and "base_image_quality" in data:
        biq = data.get("base_image_quality") or {}
        if biq.get("base_image_human_preference_score") in (None, "", "null"):
            # Try to pull base path from top-level meta (singlepass), else any turn's meta
            base_img_path = None
            if isinstance(data.get("meta"), dict):
                base_img_path = data["meta"].get("base_image_path")
            if not base_img_path:
                # multipass aggregated file: seek any turn meta
                for k, v in data.items():
                    if isinstance(v, dict) and isinstance(v.get("meta"), dict):
                        base_img_path = v["meta"].get("base_image_path")
                        if base_img_path:
                            break
            if base_img_path:
                rp = _resolve_path(repo_root, base_img_path)
                if rp:
                    base_hps_missing = True
                    image_paths.append(rp)
                    mapping["base"] = ("base_image_quality", "base_image_human_preference_score")

    # Singlepass: top-level human_preference_score
    if isinstance(data, dict) and data.get("human_preference_score") in (None, "", "null"):
        meta = data.get("meta") or {}
        tpath = meta.get("target_image_path")
        rp = _resolve_path(repo_root, tpath) if tpath else None
        if rp:
            image_paths.append(rp)
            mapping["singlepass"] = (".", "human_preference_score")

    # Multipass turns (aggregated): keys like "1", "2", ... each is a dict
    for k, v in (data.items() if isinstance(data, dict) else []):
        if not (isinstance(k, str) and k.isdigit() and isinstance(v, dict)):
            continue
        if v.get("human_preference_score") in (None, "", "null"):
            meta = v.get("meta") or {}
            tpath = meta.get("target_image_path")
            rp = _resolve_path(repo_root, tpath) if tpath else None
            if rp:
                image_paths.append(rp)
                mapping[f"turn:{k}"] = (k, "human_preference_score")

    # Deduplicate keeping order
    seen = set()
    image_paths_dedup = []
    for p in image_paths:
        if p not in seen:
            seen.add(p)
            image_paths_dedup.append(p)

    return image_paths_dedup, mapping


def _set_value_in_data(data: dict, where: Tuple[str, str], value: float):
    key, field = where
    if key == ".":
        data[field] = value
    elif key == "base":
        if "base_image_quality" in data and isinstance(data["base_image_quality"], dict):
            data["base_image_quality"][field] = value
    else:
        # turn key
        if key in data and isinstance(data[key], dict):
            data[key][field] = value


def _worker(device_token: Optional[str], files: List[Path], repo_root: Path, batch_size: int = 8, dry_run: bool = False):
    """
    device_token: string to set in CUDA_VISIBLE_DEVICES (e.g., '0' or '3'), or None for CPU.
    After setting CUDA_VISIBLE_DEVICES, always use device 'cuda:0' inside the process.
    """
    worker_name = f"GPU {device_token}" if device_token is not None else "CPU"
    if device_token is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_token)
    try:
        from hpsv3 import HPSv3RewardInferencer  # type: ignore
    except Exception as e:
        print(f"[{worker_name}] Failed to import hpsv3: {e}")
        return

    if device_token is not None:
        device_str = "cuda:0"  # within this process, the only visible GPU is index 0
    else:
        device_str = "cpu"
    try:
        infer = HPSv3RewardInferencer(device=device_str)
    except Exception as e:
        print(f"[{worker_name}] Failed to initialize HPSv3 on {device_str}: {e}")
        return

    files_total = len(files)
    print(f"[{worker_name}] Starting with {files_total} file(s).", flush=True)
    for idx, jp in enumerate(files, 1):
        print(f"[{worker_name}] {idx}/{files_total} Processing: {jp.relative_to(repo_root)}", flush=True)
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[{worker_name}] Failed to load {jp}: {e}")
            continue

        img_paths, mapping = _gather_tasks_for_file(repo_root, jp)
        if not img_paths:
            print(f"[{worker_name}] No missing HPS fields in {jp.name}", flush=True)
            continue

        # Batch through images for this file
        scores: List[float] = []
        for i in range(0, len(img_paths), batch_size):
            batch_imgs = img_paths[i:i+batch_size]
            prompts = [""] * len(batch_imgs)
            try:
                rewards = infer.reward(image_paths=batch_imgs, prompts=prompts)
                # rewards can be tensor or list-like; convert to mu float
                for r in rewards:
                    if hasattr(r, "numel"):
                        # torch tensor-like
                        try:
                            mu = r[0].item() if r.numel() >= 1 else r.item()
                        except Exception:
                            mu = float(r)
                        scores.append(float(mu))
                    elif isinstance(r, (list, tuple)) and len(r) > 0:
                        v = r[0]
                        try:
                            scores.append(float(v.item()))
                        except Exception:
                            scores.append(float(v))
                    else:
                        scores.append(float(r))
            except Exception as e:
                print(f"[{worker_name}] HPS infer failed for {jp} ({len(batch_imgs)} images): {e}")
                scores.extend([None] * len(batch_imgs))

        # Write back (match order in img_paths)
        # Build reverse lookup from path to computed score (first occurrence wins)
        path_to_score: Dict[str, Optional[float]] = {}
        for p, s in zip(img_paths, scores):
            if p not in path_to_score:
                path_to_score[p] = s

        # Update fields
        updates_written = 0
        for tag, where in mapping.items():
            # Need image path again for this tag
            # re-derive image path
            if tag == "base":
                # find base path again
                base_path = None
                if isinstance(data.get("meta"), dict):
                    base_path = data["meta"].get("base_image_path")
                if not base_path:
                    for k, v in data.items():
                        if isinstance(v, dict) and isinstance(v.get("meta"), dict):
                            base_path = v["meta"].get("base_image_path")
                            if base_path:
                                break
                rp = _resolve_path(repo_root, base_path) if base_path else None
            elif tag == "singlepass":
                tpath = (data.get("meta") or {}).get("target_image_path")
                rp = _resolve_path(repo_root, tpath) if tpath else None
            else:
                turn_key = tag.split(":", 1)[1]
                v = data.get(turn_key) or {}
                tpath = (v.get("meta") or {}).get("target_image_path")
                rp = _resolve_path(repo_root, tpath) if tpath else None

            if not rp:
                continue
            s = path_to_score.get(rp)
            if s is None:
                continue
            # Check if value would change; count updates
            key, field = where
            cur_val = None
            if key == ".":
                cur_val = data.get(field)
            elif key == "base":
                if "base_image_quality" in data and isinstance(data["base_image_quality"], dict):
                    cur_val = data["base_image_quality"].get(field)
            else:
                if key in data and isinstance(data[key], dict):
                    cur_val = data[key].get(field)
            if cur_val is None or cur_val != s:
                _set_value_in_data(data, where, float(s))
                updates_written += 1

        if not dry_run:
            try:
                with open(jp, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[{worker_name}] Failed to write {jp}: {e}")
        print(f"[{worker_name}] Updated {updates_written} field(s) in {jp.name}", flush=True)


def _chunk_list(lst: List[Path], n: int) -> List[List[Path]]:
    return [lst[i::n] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Update human_preference_score using HPSv3 across GPUs")
    parser.add_argument("--results_root", type=str, default="./evaluate_results/flux_max", help="Root folder with evaluation JSONs")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU call")
    parser.add_argument("--dry_run", action="store_true", help="Do not write changes, just simulate")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    results_root = (repo_root / args.results_root).resolve()
    if not results_root.exists():
        print(f"Results root not found: {results_root}")
        return

    # Discover JSON files that appear to need updates (quick scan)
    all_jsons = sorted([p for p in results_root.rglob("*.json")])
    if not all_jsons:
        print(f"No JSON files found under {results_root}")
        return

    # Filter to those with missing HPS somewhere
    to_process: List[Path] = []
    for jp in all_jsons:
        try:
            with open(jp, "r") as f:
                d = json.load(f)
        except Exception:
            continue
        needs = False
        if isinstance(d, dict):
            if d.get("human_preference_score") in (None, "", "null"):
                needs = True
            biq = d.get("base_image_quality") or {}
            if biq.get("base_image_human_preference_score") in (None, "", "null"):
                needs = True
            else:
                # check multipass turns
                for k, v in d.items():
                    if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
                        if v.get("human_preference_score") in (None, "", "null"):
                            needs = True
                            break
        if needs:
            to_process.append(jp)

    if not to_process:
        print("No files need updates; all HPS fields present.")
        return

    # Determine number of GPUs available
    # Determine visible GPU tokens from env if present, else 0..avail-1
    try:
        import torch
        avail = torch.cuda.device_count()
    except Exception:
        avail = 0

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_env is not None:
        tokens = [t.strip() for t in visible_env.split(",") if t.strip() != ""]
        # If env is set, the logical device_count() equals len(tokens), but we keep tokens order
        if avail > 0 and len(tokens) != avail:
            # fall back to logical indices if mismatch
            tokens = [str(i) for i in range(avail)]
    else:
        tokens = [str(i) for i in range(avail)] if avail > 0 else []

    if avail == 0:
        print("Warning: No CUDA devices detected. Attempting CPU (may be very slow).")
        tokens = []

    # Limit to requested number
    if args.num_gpus > 0 and len(tokens) > args.num_gpus:
        tokens = tokens[:args.num_gpus]

    worker_count = max(1, len(tokens))
    shards = _chunk_list(to_process, worker_count)
    print(f"Updating {len(to_process)} JSON files using {worker_count} worker(s).")

    # Spawn workers: one per token, or a single CPU worker
    procs: List[mp.Process] = []
    if tokens:
        for dev_token, files in zip(tokens, shards):
            if not files:
                continue
            p = mp.Process(target=_worker, args=(dev_token, files, repo_root, args.batch_size, args.dry_run))
            p.start()
            procs.append(p)
    else:
        # CPU single worker
        files = shards[0] if shards else []
        if files:
            p = mp.Process(target=_worker, args=(None, files, repo_root, args.batch_size, args.dry_run))
            p.start()
            procs.append(p)
    for p in procs:
        p.join()

    print("Done updating human_preference_score fields.")


if __name__ == "__main__":
    main()
