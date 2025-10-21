"""
Self-contained GPT-4o baseline generator.

Reads oai_instruction_generation_output.csv and input_images_resize_512.zip,
then generates edited images. Supports:
- Multipass: apply each instruction sequentially (turn_1, turn_2, turn_3).
- Singlepass: always use the raw image; generate turn_2 from instructions[:2] and turn_3 from instructions[:3] (skip [:1]).

Usage examples:
  python baseline_generate/gpt4o.py                          # process all (turns == 3)
  python baseline_generate/gpt4o.py --workers 4              # process with 4 threads
  python baseline_generate/gpt4o.py --csv path.csv           # custom CSV path
  python baseline_generate/gpt4o.py --zip images.zip         # custom ZIP path
  python baseline_generate/gpt4o.py --out-1024 DIR           # save originals before 512 resize
  python baseline_generate/gpt4o.py --list-incomplete        # print incomplete image_index values and exit
  python baseline_generate/gpt4o.py --only-incomplete        # process only tasks missing some outputs
  OPENAI_API_KEY=... python baseline_generate/gpt4o.py       # provide API key via env

Notes:
- No hardcoded credentials. Uses OPENAI_API_KEY env var or --api-key.
- Output (512): GPT4o_generation/{multipass,singlepass}
- Output (fullres): GPT_4o_generation_1024/{multipass,singlepass}
- Image naming: {image_index}_input_raw.png and per-turn outputs.
 - Rate limiting: 10 RPM (6 seconds) enforced globally across threads.
"""

from __future__ import annotations

import argparse
import ast
import base64
import os
import sys
import tempfile
import io
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import shutil
from threading import Lock

import pandas as pd
from PIL import Image

try:
    # New OpenAI SDK style
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


# Model/tool configuration (kept minimal and explicit)
MODEL_ID = "gpt-4.1-mini"
TOOLS_CONFIG = [{"type": "image_generation", "size": "1024x1024"}]
REQUESTS_PER_MINUTE = 5
RATE_LIMIT_DELAY = 60.0 / REQUESTS_PER_MINUTE  # 6 seconds between requests, but make it to 10 seconds.


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _save_512_from_fullres(fullres_path: str, small_path: str) -> None:
    try:
        with Image.open(fullres_path) as img:
            if img.size == (1024, 1024):
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                img.save(small_path, "PNG")
            else:
                img.save(small_path, "PNG")
    except Exception:
        # Non-fatal
        pass


class GPT4oEditor:
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_ID):
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def edit_once(
        self,
        instruction: str,
        input_image_path: str,
        output_path: str,
        fullres_output_path: Optional[str] = None,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], bool]:
        """Run a single edit call. Returns (success, response_id, moderation_blocked)."""
        try:
            image_b64 = _encode_image_base64(input_image_path)

            # Global rate limiting across threads
            _respect_rate_limit()

            if previous_response_id:
                # Follow-up turn in a multipass chain
                resp = self.client.responses.create(
                    model=self.model,
                    previous_response_id=previous_response_id,
                    input=instruction,
                    tools=TOOLS_CONFIG,
                )
            else:
                # Initial turn with image + instruction
                resp = self.client.responses.create(
                    model=self.model,
                    tools=TOOLS_CONFIG,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": instruction},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "low",
                                },
                            ],
                        }
                    ],
                )

            # Extract base64 image generation result
            image_payloads = [
                out.result for out in getattr(resp, "output", []) if getattr(out, "type", "") == "image_generation_call"
            ]
            if not image_payloads:
                return False, None, False

            img_b64 = image_payloads[0]
            img_bytes = base64.b64decode(img_b64)

            if fullres_output_path:
                # Save full resolution/original first
                with open(fullres_output_path, "wb") as f:
                    f.write(img_bytes)
                # Then save 512 variant into output_path
                _save_512_from_fullres(fullres_output_path, output_path)
            else:
                # Save directly to 512 path without leaving extra files
                try:
                    from io import BytesIO
                    with Image.open(BytesIO(img_bytes)) as im:
                        if im.size == (1024, 1024):
                            im = im.resize((512, 512), Image.Resampling.LANCZOS)
                        im.save(output_path, "PNG")
                except Exception:
                    # Fallback: write raw bytes
                    with open(output_path, "wb") as f:
                        f.write(img_bytes)
            return True, getattr(resp, "id", None), False

        except Exception as e:
            print(f"Error during image edit: {e}")
            # Heuristic: detect moderation block from error message
            et = str(e).lower()
            blocked = ("moderation_blocked" in et) or ("safety system" in et and "rejected" in et)
            return False, None, blocked


@dataclass
class RowData:
    image_index: int
    instructions: List[str]


# Simple global rate limiter (10 RPM)
_rate_lock = Lock()
_last_request_time = 0.0


def _respect_rate_limit() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        _last_request_time = time.time()


# Persist indices that were blocked by moderation so they can be skipped in future runs
_blocked_io_lock = Lock()
_BLOCKLIST_FILE = "moderation_blocked_image_indices.txt"


def _blocked_file_path(base_out_dir: str) -> str:
    return os.path.join(base_out_dir, _BLOCKLIST_FILE)


def _load_blocked_indices(base_out_dir: str) -> Set[int]:
    path = _blocked_file_path(base_out_dir)
    blocked: Set[int] = set()
    if not os.path.exists(path):
        return blocked
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    blocked.add(int(line))
                except Exception:
                    pass
    except Exception:
        pass
    return blocked


def _append_blocked_index(base_out_dir: str, image_index: int) -> None:
    # Append with a lock to avoid interleaving lines
    _ensure_dir(base_out_dir)
    path = _blocked_file_path(base_out_dir)
    try:
        with _blocked_io_lock:
            with open(path, "a") as f:
                f.write(f"{image_index}\n")
    except Exception:
        pass


def _parse_instructions(cell: str) -> List[str]:
    try:
        v = ast.literal_eval(cell)
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(cell)]
    except Exception:
        return [str(cell)]


def _find_image_in_zip(image_index: int, zip_names: List[str]) -> Optional[str]:
    base = f"{image_index}_input_raw"
    for ext in (".jpg", ".png"):
        cand = f"{base}{ext}"
        if cand in zip_names:
            return cand
    return None


def process_one_image_multipass(
    row: RowData,
    zip_path: str,
    out_dir: str,
    editor: GPT4oEditor,
    zip_names: List[str],
    out1024_dir: Optional[str] = None,
) -> str:
    try:
        entry = _find_image_in_zip(row.image_index, zip_names)
        if not entry:
            return f"❌ Image not found for index {row.image_index}"

        # Extract to temp file
        with zipfile.ZipFile(zip_path, "r") as zf, tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(zf.read(entry))
            tmp_path = tmp.name

        base = f"{row.image_index}_input_raw"
        mp_dir = os.path.join(out_dir, "multipass")
        sp_dir = os.path.join(out_dir, "singlepass")
        _ensure_dir(mp_dir)
        _ensure_dir(sp_dir)
        mp1024_dir = os.path.join(out1024_dir, "multipass") if out1024_dir else None
        sp1024_dir = os.path.join(out1024_dir, "singlepass") if out1024_dir else None
        if mp1024_dir:
            _ensure_dir(mp1024_dir)
        if sp1024_dir:
            _ensure_dir(sp1024_dir)

        # Save source image in both 1024 (full) and 512 variants
        src_mp_png = os.path.join(mp_dir, f"{base}.png")
        src_sp_png = os.path.join(sp_dir, f"{base}.png")
        src_mp_full = os.path.join(mp1024_dir, f"{base}.png") if mp1024_dir else None
        src_sp_full = os.path.join(sp1024_dir, f"{base}.png") if sp1024_dir else None
        with Image.open(tmp_path) as im:
            im = im.convert("RGB")
            # Save fullres first
            if src_mp_full:
                try:
                    im.save(src_mp_full, "PNG")
                except Exception:
                    src_mp_full = None
            if src_sp_full:
                try:
                    im.save(src_sp_full, "PNG")
                except Exception:
                    src_sp_full = None
            # Save 512 variant
            if im.size == (1024, 1024):
                im_small = im.resize((512, 512), Image.Resampling.LANCZOS)
            else:
                im_small = im
            im_small.save(src_mp_png, "PNG")
            try:
                im_small.save(src_sp_png, "PNG")
            except Exception:
                pass

        results = []
        prev_id: Optional[str] = None
        current_image_path = tmp_path  # initial turn uses the source image

        for turn, instr in enumerate(row.instructions, start=1):
            out_name = f"{base}_turn_{turn}.png"
            out_path = os.path.join(mp_dir, out_name)
            out_full = os.path.join(mp1024_dir, out_name) if mp1024_dir else None

            ok, resp_id, blocked = editor.edit_once(
                instruction=instr,
                input_image_path=current_image_path,
                output_path=out_path,
                fullres_output_path=out_full,
                previous_response_id=prev_id,
            )

            if not ok:
                if blocked:
                    _append_blocked_index(out_dir, row.image_index)
                results.append(f"❌ Failed at turn {turn}: {instr[:60]}")
                break

            results.append(f"✅ Turn {turn} -> multipass/{out_name}")
            prev_id = resp_id
            current_image_path = out_path  # next turn uses previous output

            # Rate limit centrally handled in edit_once

        # Singlepass: Use the RAW image as input.
        try:
            # Save singlepass turn_1 as a copy of multipass turn_1
            mp_t1 = os.path.join(mp_dir, f"{base}_turn_1.png")
            sp_t1 = os.path.join(sp_dir, f"{base}_turn_1.png")
            if os.path.exists(mp_t1):
                try:
                    shutil.copy2(mp_t1, sp_t1)
                    results.append(f"✅ Singlepass turn 1 -> singlepass/{os.path.basename(sp_t1)}")
                except Exception:
                    results.append("⚠️ Singlepass turn 1 copy failed")
            # And copy fullres turn_1 if available
            if mp1024_dir and sp1024_dir:
                mp_t1_full = os.path.join(mp1024_dir, f"{base}_turn_1.png")
                sp_t1_full = os.path.join(sp1024_dir, f"{base}_turn_1.png")
                if os.path.exists(mp_t1_full):
                    try:
                        shutil.copy2(mp_t1_full, sp_t1_full)
                    except Exception:
                        pass

            if len(row.instructions) >= 2:
                # Progressive prompts from the start, but skip [:1]
                max_turn = min(3, len(row.instructions))
                for upto in range(2, max_turn + 1):
                    parts = [i.strip().rstrip(".") for i in row.instructions[:upto] if str(i).strip()]
                    combined = (". ".join(parts) + ".") if parts else ""
                    sp_out = os.path.join(sp_dir, f"{base}_turn_{upto}.png")
                    sp_out_full = os.path.join(sp1024_dir, f"{base}_turn_{upto}.png") if sp1024_dir else None
                    ok, _, blocked = editor.edit_once(
                        instruction=combined,
                        input_image_path=tmp_path,  # raw image as input
                        output_path=sp_out,
                        fullres_output_path=sp_out_full,
                        previous_response_id=None,
                    )
                    if ok:
                        results.append(f"✅ Singlepass turn {upto} -> singlepass/{os.path.basename(sp_out)}")
                    else:
                        if blocked:
                            _append_blocked_index(out_dir, row.image_index)
                        results.append(f"❌ Singlepass turn {upto} failed")
                    # Rate limit centrally handled in edit_once
            else:
                results.append("ℹ️ Singlepass skipped (only 1 instruction)")
        except Exception as e:
            results.append(f"❌ Singlepass error: {e}")

        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

        return f"Image {row.image_index}: " + "; ".join(results)

    except Exception as e:
        return f"❌ Error processing image {row.image_index}: {e}"


def run_multipass(
    csv_path: str,
    zip_path: str,
    out_dir: str,
    api_key: Optional[str],
    workers: int,
    out1024_dir: Optional[str] = None,
) -> None:
    _ensure_dir(out_dir)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "turns" not in df.columns or "instructions" not in df.columns or "image_index" not in df.columns:
        raise RuntimeError("CSV must contain 'image_index', 'turns', and 'instructions' columns")

    df = df[df["turns"] == 3].copy()
    # Optional: resume/retry only incomplete tasks based on existing outputs.
    # A task is considered complete if all 3 multipass turn PNGs exist:
    #   {image_index}_input_raw_turn_{i}.png for i in {1,2,3}
    def _list_incomplete_image_indices(df_in: pd.DataFrame, base_out_dir: str, blocked: Optional[Set[int]] = None) -> list[int]:
        mp_dir = os.path.join(base_out_dir, "multipass")
        missing: list[int] = []
        for _, rr in df_in.iterrows():
            try:
                idx = int(rr["image_index"])
            except Exception:
                continue
            if blocked and idx in blocked:
                continue
            base = f"{idx}_input_raw"
            required = [
                os.path.join(mp_dir, f"{base}_turn_1.png"),
                os.path.join(mp_dir, f"{base}_turn_2.png"),
                os.path.join(mp_dir, f"{base}_turn_3.png"),
            ]
            if not all(os.path.exists(p) for p in required):
                missing.append(idx)
        return missing
    # Users sometimes want to only process tasks that are incomplete.
    # Honor environment flags to keep CLI stable without new required args.
    only_incomplete = os.environ.get("ONLY_INCOMPLETE", "0").lower() in {"1", "true", "yes"}
    list_incomplete_only = os.environ.get("LIST_INCOMPLETE_ONLY", "0").lower() in {"1", "true", "yes"}
    if only_incomplete or list_incomplete_only:
        blocked = _load_blocked_indices(out_dir)
        if blocked:
            print(f"Loaded {len(blocked)} moderation-blocked indices; skipping them.")
        incomplete_indices = _list_incomplete_image_indices(df, out_dir, blocked)
        if list_incomplete_only:
            print(f"Found {len(incomplete_indices)} incomplete image_index values:")
            if incomplete_indices:
                print(", ".join(str(i) for i in incomplete_indices))
            return
        if incomplete_indices:
            df = df[df["image_index"].isin(incomplete_indices)].copy()
        else:
            print("All tasks appear complete; nothing to do.")
            return
    print(f"Rows with turns == 3: {len(df)}")
    if df.empty:
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"Found {len(names)} files in ZIP")

    editor = GPT4oEditor(api_key=api_key)

    # Threaded over images; within each image, turns are sequential
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _, r in df.iterrows():
            row = RowData(
                image_index=int(r["image_index"]),
                instructions=_parse_instructions(str(r["instructions"])),
            )
            tasks.append(ex.submit(process_one_image_multipass, row, zip_path, out_dir, editor, names, out1024_dir))

        done = 0
        for fut in as_completed(tasks):
            try:
                msg = fut.result()
                print(msg)
            except Exception as e:
                print(f"❌ Worker error: {e}")
            finally:
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"Progress: {done}/{len(tasks)}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Self-contained GPT-4o baseline generator")
    p.add_argument("--csv", default="oai_instruction_generation_output.csv", help="Path to CSV input")
    p.add_argument("--zip", default="input_images_resize_512.zip", help="Path to ZIP with images")
    p.add_argument("--out", default="./baseline_generations/GPT4o_generation", help="Output directory")
    p.add_argument("--out-1024", default="./baseline_generations/GPT4o_generation_1024", help="Directory to save originals before 512 resize")
    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--workers", type=int, default=5, help="Number of concurrent image workers")
    p.add_argument("--only-incomplete", action="store_true", help="Process only tasks missing some outputs (multipass turns 1-3)")
    p.add_argument("--list-incomplete", action="store_true", help="List incomplete tasks and exit")

    args = p.parse_args(argv)

    print("=" * 80)
    print("GPT-4O BASELINE GENERATION (multipass)")
    print("=" * 80)
    # Bridge CLI flags to env vars used inside run_multipass to keep the function signature minimal.
    if args.only_incomplete:
        os.environ["ONLY_INCOMPLETE"] = "1"
    if args.list_incomplete:
        os.environ["LIST_INCOMPLETE_ONLY"] = "1"

    run_multipass(
        csv_path=args.csv,
        zip_path=args.zip,
        out_dir=args.out,
        api_key=args.api_key,
        workers=args.workers,
        out1024_dir=args.out_1024,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python baseline_generate/gpt4o.py --list-incomplete
# python baseline_generate/gpt4o.py --only-incomplete
