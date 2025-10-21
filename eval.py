import sys
import os
import csv
import json
import ast
import pandas as pd
import argparse
from tabulate import tabulate
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.instruction_detector import evaluate_instruction_following, load_instruction_model
from detector.quality_detector import evaluate_quality, load_quality_model
from detector.consistency_detector import evaluate_consistency, load_consistency_model
from detector.utils import _make_json_serializable

# Configuration constants
MAX_TURN = 3

# Task type definitions
LOCAL_TASK_TYPES = [
    "subject_replace", "subject_remove", "material_alter", "color_alter", 
    "subject_add", "text_change", "position_change", "count_change"
]
GLOBAL_TASK_TYPES = ["background_change"]
CONSISTENCY_TASK_TYPES = [
    "object_dinov3_consistency", "object_l1_consistency", "background_l1_consistency"
]
QUALITY_TASK_TYPES = [
    "plausibility",
    "aesthetics",
    "human_preference_score",
    "mean_s",
    "high_sat_ratio",
    "sharpness",
    "contrast",
    # Overexposure metrics
    "OEI",
    "clip_frac",
    "multi_channel_clip_frac",
    "blown_flat_frac",
    "p99",
    "p999",
    "tail_std",
]
TASK_TYPES = LOCAL_TASK_TYPES + GLOBAL_TASK_TYPES + CONSISTENCY_TASK_TYPES + QUALITY_TASK_TYPES


@dataclass
class EvaluationModels:
    """Container for all evaluation models."""
    grounding_model: object
    vlm_model: object
    quality_model: object
    dinov3_model: object
    hps_inferencer: object


@dataclass
class ImagePaths:
    """Container for image file paths."""
    base: str
    source: str
    target: str


class TaskRateCollector:
    """Manages collection and storage of task-specific rates."""
    
    def __init__(self, max_turn: int):
        self.data = {"overall": []}
        for turn in range(1, max_turn + 1):
            self.data[turn] = {task_type: [] for task_type in TASK_TYPES}
    
    def add_score(self, turn: int, task_type: str, score: float):
        """Add a score for a specific turn and task type."""
        if score is not None:
            # Be robust to newly added task types
            if task_type not in self.data[turn]:
                self.data[turn][task_type] = []
            self.data[turn][task_type].append(score)
            if task_type in LOCAL_TASK_TYPES + GLOBAL_TASK_TYPES:
                self.data["overall"].append(score)
    
    def add_quality_scores(self, turn: int, quality_results: Dict):
        """Add all quality scores for a turn."""
        for metric in QUALITY_TASK_TYPES:
            self.add_score(turn, metric, quality_results[metric])
    
    def add_consistency_scores(self, turn: int, object_result: Dict, background_result: Dict):
        """Add all consistency scores for a turn."""
        self.add_score(turn, "object_dinov3_consistency", object_result["object_dinov3_consistency_mean"])
        self.add_score(turn, "object_l1_consistency", object_result["object_l1_consistency_mean"])
        self.add_score(turn, "background_l1_consistency", background_result["bg_l1_consistency"])
        self.add_score(turn, "background_dinov3_consistency", background_result.get("bg_dinov3_masked_similarity"))


def collect_images_index(generation_folder: str, max_turn: int, mode: str = "multipass") -> List[int]:
    """Collect valid image indices that have all required files."""
    mode_folder = os.path.join(generation_folder, mode)
    if not os.path.exists(mode_folder):
        print(f"Warning: Mode folder {mode_folder} does not exist")
        return []
    
    all_files = set(os.listdir(mode_folder))
    valid_indices = []
    
    # Find all indices with raw input files (both .png and .jpg)
    for filename in all_files:
        if filename.endswith('_input_raw.png') or filename.endswith('_input_raw.jpg'):
            try:
                if filename.endswith('_input_raw.png'):
                    index = int(filename.split('_input_raw.png')[0])
                else:  # .jpg
                    index = int(filename.split('_input_raw.jpg')[0])
                
                # Check if all turn files exist for this index (turn files are always .png)
                required_files = [f"{index}_input_raw_turn_{turn}.png" for turn in range(1, max_turn + 1)]
                if all(f in all_files for f in required_files):
                    valid_indices.append(index)
            except ValueError:
                continue
    
    return sorted(valid_indices)


def get_multipass_image_paths(mode_folder: str, index: int, current_turn: int) -> ImagePaths:
    """Multipass: Generate prev->current image file paths for evaluation."""
    # Check which extension the base image has (.png or .jpg)
    base_path_png = os.path.join(mode_folder, f"{index}_input_raw.png")
    base_path_jpg = os.path.join(mode_folder, f"{index}_input_raw.jpg")
    
    if os.path.exists(base_path_png):
        base_path = base_path_png
    elif os.path.exists(base_path_jpg):
        base_path = base_path_jpg
    else:
        # Default to .png if neither exists (for consistency with original behavior)
        base_path = base_path_png
    
    source_path = (base_path if current_turn == 1 
                  else os.path.join(mode_folder, f"{index}_input_raw_turn_{current_turn-1}.png"))
    target_path = os.path.join(mode_folder, f"{index}_input_raw_turn_{current_turn}.png")
    
    return ImagePaths(base=base_path, source=source_path, target=target_path)


def parse_multipass_instruction_data(instruction_row: pd.Series, index: int, turn: int) -> Optional[Dict]:
    """Multipass: parse only the last instruction of the list for this turn."""
    try:
        format_instructions = ast.literal_eval(instruction_row["format_instructions"])
        instructions = ast.literal_eval(instruction_row["instructions"])
        task_types = ast.literal_eval(instruction_row["task_type"])
        unchanged_objects = ast.literal_eval(instruction_row["unchanged_objects"])
        all_objects = ast.literal_eval(instruction_row["all_objects"])

        if not (isinstance(format_instructions, list) and isinstance(instructions, list) and
                isinstance(task_types, list) and len(format_instructions) >= turn):
            raise ValueError("Invalid instruction data structure")

        return {
            "format_instruction": format_instructions[-1],
            "instruction": instructions[-1],
            "task_type": task_types[-1],
            "unchanged_objects": unchanged_objects,
            "all_objects": all_objects,
            "eval_bg_consistency": bool(instruction_row.get("bg_consistency", False))
        }

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing instruction data for index {index}, turn {turn}: {e}")
        return None


def parse_singlepass_instruction_sequence(instruction_row: pd.Series, index: int, turn: int) -> Optional[Dict]:
    """Singlepass: return instruction lists up to the given turn."""
    try:
        format_instructions = ast.literal_eval(instruction_row["format_instructions"])
        instructions = ast.literal_eval(instruction_row["instructions"])
        task_types = ast.literal_eval(instruction_row["task_type"])
        unchanged_objects = ast.literal_eval(instruction_row["unchanged_objects"])
        all_objects = ast.literal_eval(instruction_row["all_objects"])

        if not (isinstance(format_instructions, list) and isinstance(instructions, list) and
                isinstance(task_types, list) and len(format_instructions) >= turn):
            raise ValueError("Invalid instruction data structure for singlepass")

        return {
            "format_instructions": format_instructions[:turn],
            "instructions": instructions[:turn],
            "task_types": task_types[:turn],
            "unchanged_objects": unchanged_objects,
            "all_objects": all_objects,
            "eval_bg_consistency": bool(instruction_row.get("bg_consistency", False))
        }

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing instruction sequence for index {index}, turn {turn}: {e}")
        return None


def evaluate_turn_multipass(models: EvaluationModels, paths: ImagePaths, instruction_data: Dict, 
                            turn: int, index: int) -> Dict:
    """Multipass: evaluate prev->current turn using the last instruction only."""
    results = {}
    
    # Instruction following evaluation
    score, reason = evaluate_instruction_following(
        paths.source, paths.target, 
        formatted_instruction=instruction_data["format_instruction"],
        instruction=instruction_data["instruction"],
        task_type=instruction_data["task_type"], 
        grounding_model=models.grounding_model,
        vlm_model=models.vlm_model, 
        output_reason=True
    )
    
    results.update({
        "instruction": instruction_data["instruction"],
        "format_instructions": instruction_data["format_instruction"],
        "task_type": instruction_data["task_type"],
        "instruction_following": score,
        "instruction_following_reason": reason
    })
    
    # Quality evaluation
    quality_result = evaluate_quality(
        paths.target,
        models.quality_model,
        prompt=instruction_data["instruction"],
        hps_inferencer=models.hps_inferencer,
    )
    results.update(quality_result)
    
    # Consistency evaluation
    if instruction_data["eval_bg_consistency"]:
        object_result, background_result = evaluate_consistency(
            paths.base, paths.target, 
            instruction_data["unchanged_objects"], 
            instruction_data["all_objects"],
            grounding_model=models.grounding_model, 
            dinov3_model=models.dinov3_model
        )
        
        results.update({
            "object_dinov3_consistency_mean": object_result["object_dinov3_consistency_mean"],
            "object_l1_consistency_mean": object_result["object_l1_consistency_mean"],
            "background_l1_consistency": background_result["bg_l1_consistency"],
            "background_dinov3_consistency": background_result.get("bg_dinov3_masked_similarity"),
            "bg_details": background_result,
            "object_details": object_result
        })
    else:
        results.update({
            "object_dinov3_consistency_mean": None,
            "object_l1_consistency_mean": None,
            "background_l1_consistency": None,
            "background_dinov3_consistency": None,
            "bg_details": None,
            "object_details": None
        })
    
    # Add metadata
    results["meta"] = {
        "instruction": instruction_data["instruction"],
        "format_instructions": instruction_data["format_instruction"],
        "task_type": instruction_data["task_type"],
        "unchanged_objects": instruction_data["unchanged_objects"],
        "all_objects": instruction_data["all_objects"],
        "base_image_path": paths.base,
        "src_image_path": paths.source,
        "target_image_path": paths.target,
        "turn": turn,
        "index": index,
        "eval_bg_consistency": instruction_data["eval_bg_consistency"],
    }
    
    return results


def evaluate_turn_singlepass(models: EvaluationModels, paths: ImagePaths, instruction_seq: Dict,
                             turn: int, index: int) -> Dict:
    """Evaluate a singlepass turn by comparing base->target for each sub-instruction up to the turn.

    Records per-instruction success and reasons, plus quality and consistency on the final target.
    """
    results: Dict = {}

    insts = instruction_seq["instructions"]
    fmt_insts = instruction_seq["format_instructions"]
    types = instruction_seq["task_types"]

    scores: List[float] = []
    reasons: List[str] = []

    # Evaluate each instruction against base->target
    for sub_idx, (fmt_inst, inst, task_t) in enumerate(zip(fmt_insts, insts, types), start=1):
        score, reason = evaluate_instruction_following(
            paths.base, paths.target,
            formatted_instruction=fmt_inst,
            instruction=inst,
            task_type=task_t,
            grounding_model=models.grounding_model,
            vlm_model=models.vlm_model,
            output_reason=True,
        )
        scores.append(score)
        reasons.append(reason)

    # Choose last instruction as canonical for single-value fields (for aggregation compatibility)
    last_inst = insts[-1]
    last_fmt = fmt_insts[-1]
    last_type = types[-1]

    results.update({
        "instructions": insts,
        "format_instructions": fmt_insts,
        "task_types": types,
        "instruction_following_list": scores,
        "instruction_following_reason_list": reasons,
        # keep these single-value keys for downstream aggregations (use last one)
        "instruction": last_inst,
        "task_type": last_type,
        "instruction_following": scores[-1],
        "instruction_following_reason": reasons[-1],
    })

    # Quality on final target
    quality_result = evaluate_quality(
        paths.target,
        models.quality_model,
        prompt=last_inst,
        hps_inferencer=models.hps_inferencer,
    )
    results.update(quality_result)

    # Consistency (base vs final target)
    if instruction_seq["eval_bg_consistency"]:
        object_result, background_result = evaluate_consistency(
            paths.base, paths.target,
            instruction_seq["unchanged_objects"],
            instruction_seq["all_objects"],
            grounding_model=models.grounding_model,
            dinov3_model=models.dinov3_model,
        )
        results.update({
            "object_dinov3_consistency_mean": object_result["object_dinov3_consistency_mean"],
            "object_l1_consistency_mean": object_result["object_l1_consistency_mean"],
            "background_l1_consistency": background_result["bg_l1_consistency"],
            "background_dinov3_consistency": background_result.get("bg_dinov3_masked_similarity"),
            "bg_details": background_result,
            "object_details": object_result,
        })
    else:
        results.update({
            "object_dinov3_consistency_mean": None,
            "object_l1_consistency_mean": None,
            "background_l1_consistency": None,
            "background_dinov3_consistency": None,
            "bg_details": None,
            "object_details": None,
        })

    # Meta
    results["meta"] = {
        "instructions": insts,
        "format_instructions": fmt_insts,
        "task_types": types,
        "unchanged_objects": instruction_seq["unchanged_objects"],
        "all_objects": instruction_seq["all_objects"],
        "base_image_path": paths.base,
        "src_image_path": paths.base,  # singlepass compares base->target
        "target_image_path": paths.target,
        "turn": turn,
        "index": index,
        "eval_bg_consistency": instruction_seq["eval_bg_consistency"],
    }

    return results


def evaluate_base_image_quality(base_image_path: str, quality_model, hps_inferencer=None) -> Optional[Dict]:
    """Evaluate base image quality if it exists."""
    if not os.path.exists(base_image_path):
        return None
    
    quality_result = evaluate_quality(base_image_path, quality_model, hps_inferencer=hps_inferencer)
    return {f"base_image_{key}": value for key, value in quality_result.items()}


def save_multipass_image_result(output_folder: str, index: int, results: Dict):
    """Multipass: save results for a single image across turns."""
    output_file = os.path.join(output_folder, f"{index}_input_raw.json")
    with open(output_file, 'w') as f:
        json.dump(_make_json_serializable(results), f, indent=2)


def save_singlepass_turn_result(output_folder: str, index: int, turn: int, results: Dict):
    """Save results for a singlepass turn as {index}_input_raw_turn_{turn}.json."""
    output_file = os.path.join(output_folder, f"{index}_input_raw_turn_{turn}.json")
    with open(output_file, 'w') as f:
        json.dump(_make_json_serializable(results), f, indent=2)


def save_aggregated_results(output_folder: str, task_rates: TaskRateCollector, image_rates: Dict):
    """Save aggregated results across all images."""
    # Save aggregated results
    task_rate_file = os.path.join(output_folder, "task_rate.json")
    with open(task_rate_file, 'w') as f:
        json.dump(_make_json_serializable(task_rates.data), f, indent=2)
    
    image_rate_file = os.path.join(output_folder, "image_rate.json")
    with open(image_rate_file, 'w') as f:
        json.dump(_make_json_serializable(image_rates), f, indent=2)


def evaluate_mode(models: EvaluationModels, instruction_df: pd.DataFrame, 
                 generation_folder: str, output_folder: str, max_turn: int, mode: str):
    """Evaluate a single mode (multipass or singlepass).

    - multipass: keep existing behavior comparing prev_turn->current with last instruction.
    - singlepass: for each turn T, compare base->target_T; evaluate each instruction up to T.
    """
    print(f"Evaluating mode: {mode}")
    mode_output_folder = os.path.join(output_folder, mode)
    os.makedirs(mode_output_folder, exist_ok=True)
    
    image_indices = collect_images_index(generation_folder, max_turn, mode)
    print(f"Found {len(image_indices)} valid image indices for mode {mode}")
    
    if not image_indices:
        print(f"No valid images found for mode {mode}, skipping...")
        return
    
    task_rates = TaskRateCollector(max_turn)
    image_rates = {}
    mode_folder = os.path.join(generation_folder, mode)
    
    for index in tqdm(image_indices):
        image_rates[index] = []
        results = {}

        # Evaluate base image quality once
        base_image_path_png = os.path.join(mode_folder, f"{index}_input_raw.png")
        base_image_path_jpg = os.path.join(mode_folder, f"{index}_input_raw.jpg")

        if os.path.exists(base_image_path_png):
            base_image_path = base_image_path_png
        elif os.path.exists(base_image_path_jpg):
            base_image_path = base_image_path_jpg
        else:
            base_image_path = base_image_path_png  # Default to .png

        base_quality = evaluate_base_image_quality(base_image_path, models.quality_model, models.hps_inferencer)
        if base_quality:
            results["base_image_quality"] = base_quality

        # Process each turn
        for current_turn in range(1, max_turn + 1):
            if mode == "multipass":
                paths = get_multipass_image_paths(mode_folder, index, current_turn)
            else:  # singlepass: always compare base->target
                target_path = os.path.join(mode_folder, f"{index}_input_raw_turn_{current_turn}.png")
                paths = ImagePaths(base=base_image_path, source=base_image_path, target=target_path)

            # Check if image files exist
            if not (os.path.exists(paths.source) and os.path.exists(paths.target)):
                print(f"Warning: Image files not found for index {index}, turn {current_turn}")
                continue

            # Get instruction data for this turn
            instruction_row = instruction_df[
                (instruction_df['image_index'] == index) &
                (instruction_df['turns'] == current_turn)
            ]

            if instruction_row.empty:
                print(f"Warning: No instruction found for index {index}, turn {current_turn}")
                continue

            if mode == "multipass":
                instruction_parsed = parse_multipass_instruction_data(instruction_row.iloc[0], index, current_turn)
                if not instruction_parsed:
                    continue
                # Evaluate turn (prev->current using last instruction)
                turn_results = evaluate_turn_multipass(models, paths, instruction_parsed, current_turn, index)
                results[current_turn] = turn_results

                # Collect rates
                image_rates[index].append(turn_results["instruction_following"])
                task_rates.add_score(current_turn, instruction_parsed["task_type"], turn_results["instruction_following"])
                task_rates.add_quality_scores(current_turn, {k: v for k, v in turn_results.items() if k in QUALITY_TASK_TYPES})
                if instruction_parsed["eval_bg_consistency"]:
                    task_rates.add_consistency_scores(current_turn, turn_results["object_details"], turn_results["bg_details"])
            else:
                # singlepass: evaluate sequence up to current_turn against base->target
                instruction_seq = parse_singlepass_instruction_sequence(instruction_row.iloc[0], index, current_turn)
                if not instruction_seq:
                    continue

                turn_results = evaluate_turn_singlepass(models, paths, instruction_seq, current_turn, index)

                # attach base quality for convenience
                if base_quality:
                    turn_results.setdefault("base_image_quality", base_quality)

                # Save per-turn JSON with all sub-instruction results
                save_singlepass_turn_result(mode_output_folder, index, current_turn, turn_results)

                # Collect aggregated rates based on the last instruction for compatibility
                image_rates[index].append(turn_results["instruction_following"])
                task_rates.add_score(current_turn, turn_results.get("task_type", None), turn_results["instruction_following"])
                task_rates.add_quality_scores(current_turn, {k: v for k, v in turn_results.items() if k in QUALITY_TASK_TYPES})
                if instruction_seq["eval_bg_consistency"]:
                    task_rates.add_consistency_scores(current_turn, turn_results["object_details"], turn_results["bg_details"])

        # Save individual image results after processing all turns
        if mode == "multipass":
            # multipass keeps the single aggregated file per index
            save_multipass_image_result(mode_output_folder, index, results)
    
    # Save aggregated results after all images are processed
    save_aggregated_results(mode_output_folder, task_rates, image_rates)
    print(f"Completed evaluation for mode {mode}")


def evaluate(instruction_file: str, generation_folder: str, output_folder: str, 
             max_turn: int, modes: List[str] = None):
    """Main evaluation function."""
    if modes is None:
        modes = ["multipass", "singlepass"]
    
    # Load models once
    grounding_model, vlm_model = load_instruction_model()
    quality_model, hps_inferencer = load_quality_model()
    dinov3_model = load_consistency_model()
    models = EvaluationModels(grounding_model, vlm_model, quality_model, dinov3_model, hps_inferencer)
    print("Successfully loaded all models.")
    
    # Load instruction data
    instruction_df = pd.read_csv(instruction_file)
    print(f"Loaded instruction file with {len(instruction_df)} rows")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Evaluate each mode
    for mode in modes:
        evaluate_mode(models, instruction_df, generation_folder, output_folder, max_turn, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate instruction generation results")
    parser.add_argument("--generation_folder", type=str, required=True, help="Path to generation folder")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["multipass", "singlepass"],
        default=["multipass"],
        help="Which generation modes to evaluate (default: multipass)",
    )

    args = parser.parse_args()
    base_name = os.path.basename(args.generation_folder.rstrip('/'))
    output_folder = f"./evaluate_results/{base_name}"
    instruction_file = "./oai_instruction_generation_output.csv"

    evaluate(instruction_file, args.generation_folder, output_folder, MAX_TURN, modes=args.modes)
