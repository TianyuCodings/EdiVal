#!/usr/bin/env python
"""
Instruction generator for EdiVal powered by the OpenAI API.

This script loads filtered grounding metadata and produces the
`oai_instruction_generation_output.csv` file expected by downstream stages.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Path configuration and fallbacks
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUNDING_DIR = REPO_ROOT / "generate_instructions" / "grounding_all_objects"
DEFAULT_INSTRUCTIONS_DIR = REPO_ROOT / "generate_instructions" / "candidate_pools"
DEFAULT_INPUT_IMAGES_DIR = REPO_ROOT / "input_images_resize_512"

# Allow environment overrides while keeping string representations for legacy code
GROUNDING_DIR = os.environ.get("EDIVAL_GROUNDING_DIR", str(DEFAULT_GROUNDING_DIR))
INSTRUCTIONS_DIR = os.environ.get("EDIVAL_INSTRUCTIONS_DIR", str(DEFAULT_INSTRUCTIONS_DIR))

MAX_TURNS = 3
POSITIONS = ["left", "right", "above", "below"]

LOCAL_TASK_TYPES = [
    "subject_replace",
    "subject_remove",
    "material_alter",
    "color_alter",
    "subject_add",
    "text_change",
    "position_change",
    "count_change",
]
GLOBAL_TASK_TYPES = ["background_change"]
TASK_TYPES = LOCAL_TASK_TYPES + GLOBAL_TASK_TYPES

# Built-in fallbacks when instruction metadata files are absent
# ---------------------------------------------------------------------------
# Base generator (shared utilities reused by the OpenAI workflow)
# ---------------------------------------------------------------------------
class ImprovedCSVGenerator:
    def __init__(self, seed: int = 12345):
        random.seed(seed)
        self.object_names = self.load_file_lines("object_names.txt")
        self.colors = self.load_file_lines("colors.txt")
        self.materials = self.load_file_lines("materials.txt")
        self.backgrounds = self.load_file_lines("backgrounds.txt")
        self.texts = self.load_file_lines("texts.txt")

    def load_file_lines(self, filename: str) -> List[str]:
        """Load lines from a candidate pool file; raise if the file is missing or empty."""
        path = Path(INSTRUCTIONS_DIR) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Candidate pool '{filename}' not found at {path}. "
                "Create it under generate_instructions/candidate_pools/ or regenerate via "
                "generate_instructions/candidate_pools/generate_objects_txt.py."
            )

        with path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            raise ValueError(
                f"Candidate pool '{filename}' is empty at {path}. "
                "Populate it or regenerate via generate_instructions/candidate_pools/generate_objects_txt.py."
            )

        return lines

    def load_json_data(self, image_index: str) -> Dict[str, Any]:
        """Load and parse JSON data for a given image index."""
        json_path = Path(GROUNDING_DIR) / f"{image_index}_input_raw.json"
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "parsed_json" not in data:
                raise KeyError(f"'parsed_json' key not found in {json_path}")
            return data["parsed_json"]
        except Exception as e:  # pragma: no cover - defensive
            raise Exception(f"Error loading JSON from {json_path}: {e}")

    def get_filtered_objects(self, json_data: Dict[str, Any]) -> List[str]:
        """Extract filtered object names from JSON data."""
        filtered_objects_str = json_data.get("Filtered All Objects", "")
        if not filtered_objects_str:
            return []
        return [obj.strip().rstrip(".") for obj in filtered_objects_str.split(".") if obj.strip()]

    def get_foreground_objects(self, json_data: Dict[str, Any]) -> List[str]:
        """Get objects that have foreground=True."""
        foreground_objects = []
        for key, value in json_data.items():
            if isinstance(value, dict) and value.get("foreground", False):
                foreground_objects.append(key)
        return foreground_objects

    def init_object_pool(self, json_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Initialize object pool with current object states."""
        object_pool: Dict[str, Dict[str, Any]] = {}
        filtered_objects = self.get_filtered_objects(json_data)

        for obj_name in filtered_objects:
            if not obj_name:  # Skip empty object names
                continue

            try:
                if obj_name in json_data and isinstance(json_data[obj_name], dict):
                    object_pool[obj_name] = json_data[obj_name].copy()
                else:
                    # Default object structure if not found in JSON
                    object_pool[obj_name] = {
                        "foreground": True,
                        "color": "unknown",
                        "material": "unknown",
                        "grounding_count": 1,
                    }
            except Exception as e:
                print(f"Error processing object '{obj_name}': {e}")
                continue

        return object_pool

    def get_available_objects_for_task(self, task_type: str, object_pool: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get objects available for specific task type from current object pool."""
        available_objects: List[str] = []

        if task_type in ["subject_remove", "subject_replace", "color_alter", "material_alter", "position_change"]:
            available_objects = list(object_pool.keys())
        elif task_type == "count_change":
            for obj_name, obj_data in object_pool.items():
                if "grounding_count" in obj_data:
                    available_objects.append(obj_name)
        elif task_type == "text_change":
            available_objects = list(object_pool.keys())

        return available_objects

    def is_task_feasible(self, task_type: str, object_pool: Dict[str, Dict[str, Any]]) -> bool:
        """Check if a task type is feasible with the current object pool."""
        if task_type in {"subject_add", "background_change"}:
            return True
        if task_type == "position_change":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            return len(available_objects) >= 2
        available_objects = self.get_available_objects_for_task(task_type, object_pool)
        return len(available_objects) > 0

    def generate_format_instruction_with_pool(
        self, task_type: str, object_pool: Dict[str, Dict[str, Any]], json_data: Dict[str, Any]
    ) -> str:
        """Generate format instruction using current object pool."""

        if task_type == "subject_add":
            object_name = random.choice(self.object_names)
            available_objects = list(object_pool.keys())

            if not available_objects:
                return f"Add [{object_name}]"

            reference_object = random.choice(available_objects)
            position = random.choice(POSITIONS)
            return f"Add [{object_name}] on the [{position}] of [{reference_object}]"

        if task_type == "subject_remove":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for removal")
            object_name = random.choice(available_objects)
            return f"Remove [{object_name}]"

        if task_type == "subject_replace":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for replacement")
            object_name = random.choice(available_objects)
            new_object = random.choice(self.object_names)
            return f"Replace [{object_name}] with [{new_object}]"

        if task_type == "material_alter":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for material alteration")
            object_name = random.choice(available_objects)

            original_material = object_pool[object_name].get("material", "") if object_name in object_pool else ""
            available_materials = (
                [m for m in self.materials if m.lower() != original_material.lower()] or self.materials
            )
            material = random.choice(available_materials)
            return f"Change the material of [{object_name}] to [{material}]"

        if task_type == "color_alter":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for color alteration")
            object_name = random.choice(available_objects)

            original_color = object_pool[object_name].get("color", "") if object_name in object_pool else ""
            available_colors = [c for c in self.colors if c.lower() != original_color.lower()] or self.colors
            color = random.choice(available_colors)
            return f"Change the color of [{object_name}] to [{color}]"

        if task_type == "position_change":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if len(available_objects) < 2:
                raise ValueError("Not enough objects available for position change")
            target_object = random.choice(available_objects)
            reference_objects = [obj for obj in available_objects if obj != target_object]
            reference_object = random.choice(reference_objects) if reference_objects else available_objects[0]
            position = random.choice(POSITIONS)
            return f"Change the position of [{target_object}] to [{position}] of [{reference_object}]"

        if task_type == "count_change":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for count change")
            object_name = random.choice(available_objects)

            original_count = object_pool[object_name].get("grounding_count", 1) if object_name in object_pool else 1
            available_counts = [c for c in [1, 2, 3, 4, 5] if c != original_count] or [1, 2, 3, 4, 5]
            count = random.choice(available_counts)
            return f"Change the count of [{object_name}] to [{count}]"

        if task_type == "background_change":
            background = random.choice(self.backgrounds)
            return f"Change the background to [{background}]"

        if task_type == "text_change":
            objects_with_text = []
            for key, value in json_data.items():
                if isinstance(value, dict) and value.get("text") is not None:
                    objects_with_text.append((key, value["text"]))

            available_objects = list(object_pool.keys())
            valid_objects_with_text = [(obj_name, text) for obj_name, text in objects_with_text if obj_name in available_objects]

            if valid_objects_with_text:
                object_name, existing_text = random.choice(valid_objects_with_text)
                new_text = random.choice(self.texts)
                return f"Replace the text '[{existing_text}]' on [{object_name}] with '[{new_text}]'"

            new_text = random.choice(self.texts)
            return f"Add text '[{new_text}]' on the image"

        return f"[{task_type}] instruction"

    def parse_format_instruction(self, task_type: str, format_instruction: str) -> Dict[str, Any]:
        """Parse format instruction and extract relevant object information using regex patterns."""
        result = {
            "edited_objects": [],
            "added_objects": [],
            "removed_objects": [],
            "modifications": {},
        }

        if task_type == "subject_add":
            match = re.search(r"Add \[([^\]]+)\]", format_instruction)
            if match:
                new_object = match.group(1)
                result["added_objects"].append(
                    {
                        "name": new_object,
                        "data": {
                            "foreground": True,
                            "color": "unknown",
                            "material": "unknown",
                            "grounding_count": 1,
                        },
                    }
                )

        elif task_type == "subject_remove":
            match = re.search(r"Remove \[([^\]]+)\]", format_instruction)
            if match:
                removed_object = match.group(1)
                result["edited_objects"].append(removed_object)
                result["removed_objects"].append(removed_object)

        elif task_type == "subject_replace":
            match = re.search(r"Replace \[([^\]]+)\] with \[([^\]]+)\]", format_instruction)
            if match:
                old_object, new_object = match.groups()
                result["edited_objects"].append(old_object)
                result["added_objects"].append({"name": new_object, "copy_from": old_object})
                result["removed_objects"].append(old_object)

        elif task_type == "color_alter":
            match = re.search(r"Change the color of \[([^\]]+)\] to \[([^\]]+)\]", format_instruction)
            if match:
                object_name, new_color = match.groups()
                result["edited_objects"].append(object_name)
                result["modifications"][object_name] = {"color": new_color}

        elif task_type == "material_alter":
            match = re.search(r"Change the material of \[([^\]]+)\] to \[([^\]]+)\]", format_instruction)
            if match:
                object_name, new_material = match.groups()
                result["edited_objects"].append(object_name)
                result["modifications"][object_name] = {"material": new_material}

        elif task_type == "count_change":
            match = re.search(r"Change the count of \[([^\]]+)\] to \[([^\]]+)\]", format_instruction)
            if match:
                object_name, new_count = match.groups()
                result["edited_objects"].append(object_name)
                try:
                    result["modifications"][object_name] = {"grounding_count": int(new_count)}
                except ValueError:
                    pass

        elif task_type == "position_change":
            match = re.search(r"Change the position of \[([^\]]+)\] to \[([^\]]+)\] of \[([^\]]+)\]", format_instruction)
            if match:
                target_object, position, reference_object = match.groups()
                result["edited_objects"].append(target_object)
                result["edited_objects"].append(reference_object)

        elif task_type == "text_change":
            replace_match = re.search(
                r"Replace the text '\[([^\]]+)\]' on \[([^\]]+)\] with '\[([^\]]+)\]'",
                format_instruction,
            )
            if replace_match:
                _, text_object, _ = replace_match.groups()
                result["edited_objects"].append(text_object)

        return result

    def update_object_pool(self, object_pool: Dict[str, Dict[str, Any]], task_type: str, format_instruction: str) -> Dict[str, Dict[str, Any]]:
        """Update object pool based on the current task."""
        new_pool = copy.deepcopy(object_pool)
        parsed = self.parse_format_instruction(task_type, format_instruction)

        for obj_name in parsed["removed_objects"]:
            new_pool.pop(obj_name, None)

        for obj_info in parsed["added_objects"]:
            obj_name = obj_info["name"]
            if "copy_from" in obj_info and obj_info["copy_from"] in object_pool:
                new_pool[obj_name] = object_pool[obj_info["copy_from"]].copy()
            elif "data" in obj_info:
                new_pool[obj_name] = obj_info["data"]
            else:
                new_pool[obj_name] = {
                    "foreground": True,
                    "color": "unknown",
                    "material": "unknown",
                    "grounding_count": 1,
                }

        for obj_name, modifications in parsed["modifications"].items():
            if obj_name in new_pool:
                new_pool[obj_name].update(modifications)

                new_obj_name = obj_name
                if "color" in modifications:
                    new_obj_name = self._create_modified_object_name(obj_name, "color", modifications["color"])
                elif "material" in modifications:
                    new_obj_name = self._create_modified_object_name(obj_name, "material", modifications["material"])

                if new_obj_name != obj_name:
                    new_pool[new_obj_name] = new_pool.pop(obj_name)

        return new_pool

    def convert_format_to_instruction(
        self, format_instruction: str, task_type: str, json_data: Dict[str, Any], available_objects_pool: Dict[str, Dict[str, Any]] | None = None
    ) -> str:
        """Convert format instruction to actual instruction by removing brackets."""
        if task_type == "background_change":
            instruction = format_instruction.replace("[", "").replace("]", "")
            if available_objects_pool is not None:
                foreground_objects = self.get_foreground_objects(available_objects_pool)
                if foreground_objects:
                    objects_str = ", ".join(foreground_objects)
                    instruction += f", remain the {objects_str} unchanged"
            return instruction
        return format_instruction.replace("[", "").replace("]", "")

    def get_edited_objects_from_instruction(self, task_type: str, format_instruction: str) -> List[str]:
        """Extract which objects are being edited from a format instruction."""
        parsed = self.parse_format_instruction(task_type, format_instruction)
        return parsed["edited_objects"]

    def get_unchanged_objects_list(self, original_objects: List[str], all_edited_objects: set[str]) -> List[str]:
        """Get objects that have remained unchanged throughout all editing turns."""
        return [obj for obj in original_objects if obj not in all_edited_objects]

    def _create_modified_object_name(self, original_name: str, modification_type: str, new_value: str) -> str:
        """Helper function to create modified object names for color/material changes."""
        if " " in original_name:
            words = original_name.split()
            attribute_list = self.colors if modification_type == "color" else self.materials
            attribute_replaced = False

            for i, word in enumerate(words):
                if any(attr.lower() in word.lower() for attr in attribute_list):
                    words[i] = new_value
                    attribute_replaced = True
                    break

            if not attribute_replaced:
                words.insert(0, new_value)

            return " ".join(words)
        return f"{new_value} {original_name}"

    def get_all_objects_list(
        self,
        all_objects_history: List[str],
        object_pool: Dict[str, Dict[str, Any]],
        task_type: str,
        format_instruction: str,
    ) -> List[str]:
        """Update all objects list including new/modified objects."""
        new_all_objects = all_objects_history.copy()
        parsed = self.parse_format_instruction(task_type, format_instruction)

        for obj_info in parsed["added_objects"]:
            obj_name = obj_info["name"]
            if obj_name not in new_all_objects:
                new_all_objects.append(obj_name)

        for obj_name, modifications in parsed["modifications"].items():
            if "color" in modifications:
                modified_name = self._create_modified_object_name(obj_name, "color", modifications["color"])
                if modified_name not in new_all_objects:
                    new_all_objects.append(modified_name)
            elif "material" in modifications:
                modified_name = self._create_modified_object_name(obj_name, "material", modifications["material"])
                if modified_name not in new_all_objects:
                    new_all_objects.append(modified_name)

        if task_type == "text_change":
            add_match = re.search(r"Add text '\[([^\]]+)\]' on the image", format_instruction)
            if add_match:
                new_text = add_match.group(1)
                if new_text not in new_all_objects:
                    new_all_objects.append(new_text)

        return new_all_objects

    def generate_csv_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate the complete CSV data with object tracking."""
        csv_data: List[Dict[str, Any]] = []
        grounding_path = Path(GROUNDING_DIR)
        if not grounding_path.exists():
            raise FileNotFoundError(f"Grounding directory not found: {grounding_path}")

        grounding_files = sorted(grounding_path.glob("*_input_raw.json"))
        image_indices = [f.name.split("_")[0] for f in grounding_files]
        if limit is not None:
            image_indices = image_indices[:limit]

        for image_index in image_indices:
            try:
                json_data = self.load_json_data(image_index)
                original_objects = self.get_filtered_objects(json_data)
                if not original_objects:
                    print(f"Skipping image {image_index}: No filtered objects found")
                    continue

                all_objects_ever = original_objects.copy()
                unchanged_objects_base = original_objects.copy()
                available_objects_pool = self.init_object_pool(json_data)

                used_task_types: List[str] = []
                has_bg_change_before = False
                all_edited_objects: set[str] = set()

                persistent_format_instructions: List[str] = []
                persistent_instructions: List[str] = []
                persistent_actual_tasks: List[str] = []

                for turn in range(1, MAX_TURNS + 1):
                    available_tasks = [task for task in TASK_TYPES if task not in used_task_types] or TASK_TYPES

                    if available_tasks:
                        new_task = random.choice(available_tasks)
                        used_task_types.append(new_task)

                        if self.is_task_feasible(new_task, available_objects_pool):
                            actual_task = new_task
                        else:
                            actual_task = "subject_add"

                        format_instr = self.generate_format_instruction_with_pool(actual_task, available_objects_pool, json_data)
                        actual_instr = self.convert_format_to_instruction(format_instr, actual_task, json_data, available_objects_pool)

                        persistent_actual_tasks.append(actual_task)
                        persistent_format_instructions.append(format_instr)
                        persistent_instructions.append(actual_instr)

                        edited_objects = self.get_edited_objects_from_instruction(actual_task, format_instr)
                        all_edited_objects.update(edited_objects)

                        available_objects_pool = self.update_object_pool(available_objects_pool, actual_task, format_instr)
                        all_objects_ever = self.get_all_objects_list(all_objects_ever, available_objects_pool, actual_task, format_instr)

                        if actual_task == "background_change":
                            has_bg_change_before = True
                            foreground_objects = self.get_foreground_objects(json_data)
                            available_objects_pool = {k: v for k, v in available_objects_pool.items() if k in foreground_objects}

                    current_turn_tasks = persistent_actual_tasks[:turn]
                    format_instructions = persistent_format_instructions[:turn]
                    instructions = persistent_instructions[:turn]

                    all_objects = all_objects_ever
                    unchanged_objects = self.get_unchanged_objects_list(unchanged_objects_base, all_edited_objects)
                    unchanged_objects = [obj for obj in unchanged_objects if obj in available_objects_pool]
                    available_objects = list(available_objects_pool.keys())
                    bg_consistency = not has_bg_change_before

                    row = {
                        "image_index": int(image_index),
                        "turns": turn,
                        "task_type": current_turn_tasks,
                        "instructions": instructions,
                        "format_instructions": format_instructions,
                        "bg_consistency": bg_consistency,
                        "unchanged_objects": unchanged_objects,
                        "available_objects": available_objects,
                        "all_objects": all_objects,
                    }

                    csv_data.append(row)

            except Exception as e:
                import traceback

                print(f"Error processing image {image_index}: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
                continue

        return csv_data

    def save_csv(self, csv_data: List[Dict[str, Any]], output_path: str) -> None:
        """Save CSV data to file."""
        df = pd.DataFrame(csv_data)
        df = df.sort_values(by=["image_index", "turns"])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"CSV saved to {output_path} with {len(csv_data)} rows")


# ---------------------------------------------------------------------------
# OpenAI-assisted generator
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"


class OpenAIInstructionGenerator(ImprovedCSVGenerator):
    def __init__(self, api_key: str, input_images_dir: str, seed: int = 42):
        super().__init__(seed=seed)
        if OpenAI is None:  # pragma: no cover - runtime check
            raise ImportError("openai package is required for instruction generation")
        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_ENDPOINT")
        )
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = str(base_url).rstrip("/")
        self.client = OpenAI(**client_kwargs)
        self.input_images_dir = input_images_dir

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string with compression."""
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=85, optimize=True)
            compressed_data = output.getvalue()
            return base64.b64encode(compressed_data).decode("utf-8")

    def get_image_path(self, image_index: str) -> Optional[str]:
        """Get the path to the image file for a given index."""
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

        for ext in extensions:
            patterns = [
                f"{image_index}_input_raw.{ext}",
                f"{image_index}.{ext}",
                f"{image_index}_input.{ext}",
                f"{image_index}_original.{ext}",
            ]

            for pattern in patterns:
                path = Path(self.input_images_dir) / pattern
                if path.exists():
                    return str(path)

        return None

    def call_openai_with_image(self, prompt: str, image_path: str) -> str:
        """Call OpenAI API with image and prompt."""
        try:
            base64_image = self.encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            response = self.client.chat.completions.create(
                messages=messages, max_tokens=100, temperature=0.7, top_p=1.0, model=MODEL_NAME
            )

            content = response.choices[0].message.content.strip()
            return content

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""

    # --- OpenAI-backed helpers -------------------------------------------------
    def generate_oai_subject_replace(self, object_name: str, image_index: str) -> str:
        image_path = self.get_image_path(image_index)
        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            return random.choice(self.object_names)

        prompt = SUBJECT_REPLACE_PROMPT.format(object_name=object_name)
        response = self.call_openai_with_image(prompt, image_path)
        response = response.strip().lower().replace("'", "").replace('"', "")
        if response:
            return response
        return random.choice(self.object_names)

    def generate_oai_material_alter(self, object_name: str, current_material: str, image_index: str) -> str:
        image_path = self.get_image_path(image_index)
        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            available_materials = [m for m in self.materials if m.lower() != current_material.lower()] or self.materials
            return random.choice(available_materials)

        prompt = MATERIAL_ALTER_PROMPT.format(
            object_name=object_name, current_material=current_material if current_material != "unknown" else "unknown"
        )
        response = self.call_openai_with_image(prompt, image_path)
        response = response.strip().lower()
        if response:
            return response
        available_materials = [m for m in self.materials if m.lower() != current_material.lower()] or self.materials
        return random.choice(available_materials)

    def generate_oai_position_change(self, available_objects: List[str], image_index: str) -> str:
        if len(available_objects) < 2:
            raise ValueError("Not enough objects for position change")

        image_path = self.get_image_path(image_index)
        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            target_object = random.choice(available_objects)
            reference_objects = [obj for obj in available_objects if obj != target_object]
            reference_object = random.choice(reference_objects)
            position = random.choice(POSITIONS)
            return f"Change the position of [{target_object}] to [{position}] of [{reference_object}]"

        prompt = POSITION_CHANGE_PROMPT.format(available_objects=", ".join(available_objects))
        response = self.call_openai_with_image(prompt, image_path)
        if "Change the position of [" in response and "] to [" in response and "] of [" in response:
            return response
        target_object = random.choice(available_objects)
        reference_objects = [obj for obj in available_objects if obj != target_object]
        reference_object = random.choice(reference_objects)
        position = random.choice(POSITIONS)
        return f"Change the position of [{target_object}] to [{position}] of [{reference_object}]"

    def generate_oai_count_change(self, available_objects: List[str], target_count: int, image_index: str) -> str:
        if not available_objects:
            raise ValueError("No objects available for count change")

        image_path = self.get_image_path(image_index)
        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            object_name = random.choice(available_objects)
            return f"Change the count of [{object_name}] to [{target_count}]"

        prompt = COUNT_CHANGE_PROMPT.format(available_objects=", ".join(available_objects), target_count=target_count)
        response = self.call_openai_with_image(prompt, image_path)
        if f"Change the count of [" in response and f"] to [{target_count}]" in response:
            return response
        object_name = random.choice(available_objects)
        return f"Change the count of [{object_name}] to [{target_count}]"

    def generate_oai_text_change(self, json_data: Dict[str, Any], available_objects: List[str], image_index: str) -> str:
        image_path = self.get_image_path(image_index)
        objects_with_text = []
        for key, value in json_data.items():
            if isinstance(value, dict) and value.get("text") is not None:
                objects_with_text.append((key, value["text"]))

        valid_objects_with_text = [(obj_name, text) for obj_name, text in objects_with_text if obj_name in available_objects]

        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            if valid_objects_with_text:
                object_name, existing_text = random.choice(valid_objects_with_text)
                new_text = random.choice(self.texts)
                return f"Replace the text '[{existing_text}]' on [{object_name}] with '[{new_text}]'"
            new_text = random.choice(self.texts)
            return f"Add text '[{new_text}]' on the image"

        if valid_objects_with_text:
            object_name, existing_text = random.choice(valid_objects_with_text)
            text_situation = f"Replace existing text '{existing_text}' on {object_name}"
            prompt = TEXT_CHANGE_PROMPT.format(text_situation=text_situation)
            response = self.call_openai_with_image(prompt, image_path)
            new_text = response.strip().replace("'", "").replace('"', "")
            if new_text:
                return f"Replace the text '[{existing_text}]' on [{object_name}] with '[{new_text}]'"
            new_text = random.choice(self.texts)
            return f"Replace the text '[{existing_text}]' on [{object_name}] with '[{new_text}]'"

        text_situation = "Add new text to the image"
        prompt = TEXT_CHANGE_PROMPT.format(text_situation=text_situation)
        response = self.call_openai_with_image(prompt, image_path)
        new_text = response.strip().replace("'", "").replace('"', "")
        if new_text:
            return f"Add text '[{new_text}]' on the image"
        new_text = random.choice(self.texts)
        return f"Add text '[{new_text}]' on the image"

    def generate_oai_color_alter(self, object_name: str, current_color: str, image_index: str) -> str:
        image_path = self.get_image_path(image_index)
        basic_colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "brown",
            "gray",
            "orange",
            "purple",
            "pink",
        ]

        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            available_colors = [c for c in basic_colors if c.lower() != current_color.lower()] or basic_colors
            return random.choice(available_colors)

        prompt = COLOR_ALTER_PROMPT.format(object_name=object_name, current_color=current_color or "unknown")
        response = self.call_openai_with_image(prompt, image_path)
        response = response.strip().lower().replace("'", "").replace('"', "")

        if response in basic_colors and response.lower() != current_color.lower():
            return response
        available_colors = [c for c in basic_colors if c.lower() != current_color.lower()] or basic_colors
        return random.choice(available_colors)

    def generate_oai_subject_add(self, reference_object: str, position: str, image_index: str) -> str:
        image_path = self.get_image_path(image_index)
        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            object_name = random.choice(self.object_names)
            return f"Add [{object_name}] on the [{position}] of [{reference_object}]"

        prompt = SUBJECT_ADD_PROMPT.format(reference_object=reference_object, position=position)
        response = self.call_openai_with_image(prompt, image_path)
        new_object = response.strip().lower().replace("'", "").replace('"', "")
        if new_object:
            return f"Add [{new_object}] on the [{position}] of [{reference_object}]"
        object_name = random.choice(self.object_names)
        return f"Add [{object_name}] on the [{position}] of [{reference_object}]"

    def generate_oai_background_change(self, image_index: str) -> str:
        image_path = self.get_image_path(image_index)
        if not image_path:
            print(f"Warning: Image not found for index {image_index}, using random selection")
            return random.choice(self.backgrounds)

        prompt = BACKGROUND_CHANGE_PROMPT
        response = self.call_openai_with_image(prompt, image_path)
        response = response.strip().lower().replace("'", "").replace('"', "")
        if response:
            return f"Change the background to [{response}]"
        return f"Change the background to [{random.choice(self.backgrounds)}]"

    # ------------------------------------------------------------------
    def generate_format_instruction_with_pool_oai(
        self, task_type: str, object_pool: Dict[str, Dict[str, Any]], json_data: Dict[str, Any], image_index: str
    ) -> str:
        """Enhanced version that uses OpenAI for specific task types."""

        if task_type == "subject_replace":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for replacement")
            object_name = random.choice(available_objects)
            new_object = self.generate_oai_subject_replace(object_name, image_index)
            return f"Replace [{object_name}] with [{new_object}]"

        if task_type == "material_alter":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for material alteration")
            object_name = random.choice(available_objects)
            original_material = object_pool.get(object_name, {}).get("material", "unknown")
            new_material = self.generate_oai_material_alter(object_name, original_material, image_index)
            return f"Change the material of [{object_name}] to [{new_material}]"

        if task_type == "position_change":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            return self.generate_oai_position_change(available_objects, image_index)

        if task_type == "count_change":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for count change")
            object_name = random.choice(available_objects)
            original_count = object_pool.get(object_name, {}).get("grounding_count", 1)
            available_counts = [c for c in [3, 4, 5] if c != original_count] or [3, 4, 5]
            target_count = random.choice(available_counts)
            return self.generate_oai_count_change(available_objects, target_count, image_index)

        if task_type == "color_alter":
            available_objects = self.get_available_objects_for_task(task_type, object_pool)
            if not available_objects:
                raise ValueError("No objects available for color alteration")
            object_name = random.choice(available_objects)
            original_color = object_pool.get(object_name, {}).get("color", "unknown")
            new_color = self.generate_oai_color_alter(object_name, original_color, image_index)
            return f"Change the color of [{object_name}] to [{new_color}]"

        if task_type == "text_change":
            available_objects = list(object_pool.keys())
            return self.generate_oai_text_change(json_data, available_objects, image_index)

        if task_type == "subject_add":
            available_objects = list(object_pool.keys())
            if not available_objects:
                image_path = self.get_image_path(image_index)
                if image_path:
                    prompt = SUBJECT_ADD_PROMPT.format(reference_object="the scene", position="in")
                    response = self.call_openai_with_image(prompt, image_path)
                    new_object = response.strip().lower().replace("'", "").replace('"', "")
                    if new_object:
                        return f"Add [{new_object}]"
                return f"Add [{random.choice(self.object_names)}]"

            reference_object = random.choice(available_objects)
            position = random.choice(POSITIONS)
            return self.generate_oai_subject_add(reference_object, position, image_index)

        if task_type == "background_change":
            return self.generate_oai_background_change(image_index)

        return self.generate_format_instruction_with_pool(task_type, object_pool, json_data)

    def generate_csv_data_with_oai(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate CSV data using OpenAI for specific task types."""
        csv_data: List[Dict[str, Any]] = []

        grounding_path = Path(self.grounding_dir)
        if not grounding_path.exists():
            raise FileNotFoundError(f"Grounding directory not found: {grounding_path}")

        grounding_files = sorted(grounding_path.glob("*_input_raw.json"))
        image_indices = [f.name.split("_")[0] for f in grounding_files]
        if limit is not None:
            image_indices = image_indices[:limit]

        oai_task_types = {
            "subject_replace",
            "material_alter",
            "position_change",
            "count_change",
            "color_alter",
            "text_change",
            "subject_add",
            "background_change",
        }

        for image_index in tqdm(image_indices, desc="Processing images with OpenAI"):
            try:
                json_data = self.load_json_data(image_index)
                original_objects = self.get_filtered_objects(json_data)

                if not original_objects:
                    print(f"Skipping image {image_index}: No filtered objects found")
                    continue

                all_objects_ever = original_objects.copy()
                unchanged_objects_base = original_objects.copy()
                available_objects_pool = self.init_object_pool(json_data)

                used_task_types: List[str] = []
                has_bg_change_before = False
                all_edited_objects: set[str] = set()

                persistent_format_instructions: List[str] = []
                persistent_instructions: List[str] = []
                persistent_actual_tasks: List[str] = []

                for turn in range(1, self.max_turns + 1):
                    available_tasks = [task for task in self.task_types if task not in used_task_types] or self.task_types

                    if available_tasks:
                        new_task = random.choice(available_tasks)
                        used_task_types.append(new_task)

                        if self.is_task_feasible(new_task, available_objects_pool):
                            actual_task = new_task
                        else:
                            actual_task = "subject_add"

                        if actual_task in oai_task_types:
                            try:
                                format_instr = self.generate_format_instruction_with_pool_oai(
                                    actual_task, available_objects_pool, json_data, image_index
                                )
                            except Exception as e:
                                print(f"OpenAI generation failed for {actual_task} in image {image_index}: {e}")
                                format_instr = self.generate_format_instruction_with_pool(
                                    actual_task, available_objects_pool, json_data
                                )
                        else:
                            format_instr = self.generate_format_instruction_with_pool(
                                actual_task, available_objects_pool, json_data
                            )

                        actual_instr = self.convert_format_to_instruction(
                            format_instr, actual_task, json_data, available_objects_pool
                        )

                        persistent_actual_tasks.append(actual_task)
                        persistent_format_instructions.append(format_instr)
                        persistent_instructions.append(actual_instr)

                        edited_objects = self.get_edited_objects_from_instruction(actual_task, format_instr)
                        all_edited_objects.update(edited_objects)

                        available_objects_pool = self.update_object_pool(available_objects_pool, actual_task, format_instr)
                        all_objects_ever = self.get_all_objects_list(all_objects_ever, available_objects_pool, actual_task, format_instr)

                        if actual_task == "background_change":
                            has_bg_change_before = True
                            foreground_objects = self.get_foreground_objects(json_data)
                            available_objects_pool = {k: v for k, v in available_objects_pool.items() if k in foreground_objects}

                    current_turn_tasks = persistent_actual_tasks[:turn]
                    format_instructions = persistent_format_instructions[:turn]
                    instructions = persistent_instructions[:turn]

                    all_objects = all_objects_ever
                    unchanged_objects = self.get_unchanged_objects_list(unchanged_objects_base, all_edited_objects)
                    unchanged_objects = [obj for obj in unchanged_objects if obj in available_objects_pool]
                    available_objects = list(available_objects_pool.keys())
                    bg_consistency = not has_bg_change_before

                    row = {
                        "image_index": int(image_index),
                        "turns": turn,
                        "task_type": current_turn_tasks,
                        "instructions": instructions,
                        "format_instructions": format_instructions,
                        "bg_consistency": bg_consistency,
                        "unchanged_objects": unchanged_objects,
                        "available_objects": available_objects,
                        "all_objects": all_objects,
                    }

                    csv_data.append(row)

            except Exception as e:
                import traceback

                print(f"Error processing image {image_index}: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
                continue

        return csv_data

    @property
    def task_types(self) -> List[str]:
        return TASK_TYPES

    @property
    def max_turns(self) -> int:
        return MAX_TURNS

    @property
    def grounding_dir(self) -> str:
        return GROUNDING_DIR


# ---------------------------------------------------------------------------
# Prompt templates (unchanged from original script)
# ---------------------------------------------------------------------------
SUBJECT_REPLACE_PROMPT = """
You are given an image and asked to suggest a replacement object for a specific object in the scene.

Given object to replace: {object_name}

Your task:
1. Look at the image and understand the context/scene
2. Suggest a new object that would fit naturally in this scene as a replacement for "{object_name}"
3. The new object should be contextually appropriate and realistic for the scene
4. Respond with ONLY the object name (e.g., "chair", "lamp", "book") - no additional text

Examples:
- If replacing a "cup" in a kitchen scene, you might suggest "bowl" or "mug"
- If replacing a "car" in a street scene, you might suggest "bus" or "truck"
- If replacing a "chair" in an office, you might suggest "stool" or "bench"

New object name:"""

MATERIAL_ALTER_PROMPT = """
You are given an image and asked to suggest a new material for a specific object.

Object to change material: {object_name}
Current material: {current_material}

Your task:
1. Look at the image and identify the object
2. Suggest a realistic alternative material for this type of object
3. The material should be different and easy to distinguish from the current material and realistic for the object type
4. Respond with ONLY the material name (e.g., "wood", "metal", "plastic", "leather") - no additional text

Examples:
- For a "cup": ceramic, glass, metal, plastic
- For a "chair": wood, metal, plastic, fabric
- For a "bag": leather, canvas, nylon, fabric

New material:"""

POSITION_CHANGE_PROMPT = """
You are given an image and asked to create a position change instruction.

Available objects in the image: {available_objects}
Position options: left, right, above, below

Your task:
1. Look at the image and identify the objects
2. Create a realistic position change instruction by selecting:
   - A target object to move
   - A reference object to position relative to
   - A position relationship (left, right, above, below)
3. The instruction should make sense given the current layout and be physically reasonable
4. Respond in this EXACT format: "Change the position of [target_object] to [position] of [reference_object]"

Examples:
- "Change the position of [cup] to [right] of [book]"
- "Change the position of [lamp] to [above] of [table]"

Position change instruction:"""

COUNT_CHANGE_PROMPT = """
You are given an image and asked to create a count change instruction.

Available objects in the image: {available_objects}
Target count: {target_count}

Your task:
1. Look at the image and identify the objects
2. Select an object that would make sense to have {target_count} instances of
3. Consider what would be realistic and natural in the scene
4. Respond in this EXACT format: "Change the count of [object_name] to [{target_count}]"

Examples:
- "Change the count of [cup] to [3]"
- "Change the count of [book] to [2]"

Count change instruction:"""

TEXT_CHANGE_PROMPT = """
You are given an image and asked to generate new text content.

Current situation: {text_situation}

Your task:
1. Look at the image and understand the context
2. Generate appropriate text that fits the scene
3. Keep text SHORT: maximum 2 words in English or 4 characters in Chinese
4. Make it contextually appropriate for the scene
5. Respond with ONLY the text content (no quotes, no additional words)

Examples:
- For a coffee shop: "COFFEE" or "OPEN"
- For a book: "NOVEL" or "GUIDE"
- For a sign: "EXIT" or "STOP"
- For Chinese: "咖啡" or "出口"

New text:"""

COLOR_ALTER_PROMPT = """
You are given an image and asked to suggest a new color for a specific object.

Object to change color: {object_name}
Current color: {current_color}

Your task:
1. Look at the image and identify the object
2. Suggest a simple, common color that would look good on this object
3. Use only basic color names: red, blue, green, yellow, black, white, brown, gray, orange, purple, pink
4. Avoid complex color names like navy, crimson, turquoise, etc.
5. Choose a color different from the current color
6. Respond with ONLY the color name (e.g., "red", "blue") - no additional text

Basic colors to choose from: red, blue, green, yellow, black, white, brown, gray, orange, purple, pink

New color:"""

SUBJECT_ADD_PROMPT = """
You are given an image and asked to suggest a new object to add to the scene.

Reference object: {reference_object}
Position: {position}
Context: Add a new object {position} of {reference_object}

Your task:
1. Look at the image and understand the scene context
2. Consider what object would naturally fit {position} of {reference_object}
3. Suggest an object that makes sense in this environment
4. Choose something that would be realistic and contextually appropriate
5. Respond with ONLY the object name (e.g., "lamp", "book", "cup") - no additional text

Examples:
- Next to a desk: "chair", "lamp", "computer"
- Near a kitchen counter: "bowl", "plate", "mug"
- By a window: "plant", "curtain", "book"

New object:"""

BACKGROUND_CHANGE_PROMPT = """
You are given an image and asked to suggest a new background for the scene.

Current scene context: The image contains various objects that should remain unchanged.

Your task:
1. Look at the image and understand the current setting
2. Suggest a new background that would work well with the existing objects
3. Keep it simple and realistic
4. Use 1-2 words maximum (e.g., "kitchen", "office", "garden", "beach", "forest")
5. Respond with ONLY the background name - no additional text

Examples:
- "kitchen"
- "office"
- "garden"
- "beach"
- "forest"
- "bedroom"

New background:"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EdiVal instruction CSVs with the OpenAI API.")
    parser.add_argument("--output", default="oai_instruction_generation_output.csv", help="Destination CSV path")
    parser.add_argument("--grounding-dir", help="Directory containing *_input_raw.json grounding files")
    parser.add_argument("--instructions-dir", help="Directory containing instruction metadata txt files")
    parser.add_argument("--input-images", help="Directory with source images (defaults to input_images_resize_512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to process")
    parser.add_argument("--api-key", default=None, help="Explicit OpenAI API key (overrides environment)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global GROUNDING_DIR, INSTRUCTIONS_DIR
    if args.grounding_dir:
        GROUNDING_DIR = str(Path(args.grounding_dir).expanduser().resolve())
    if args.instructions_dir:
        INSTRUCTIONS_DIR = str(Path(args.instructions_dir).expanduser().resolve())

    random.seed(args.seed)

    if load_dotenv is not None:
        load_dotenv()
    api_key = (
        args.api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY before running."
        )

    input_dir = args.input_images or DEFAULT_INPUT_IMAGES_DIR
    generator = OpenAIInstructionGenerator(api_key=api_key, input_images_dir=str(input_dir), seed=args.seed)
    csv_data = generator.generate_csv_data_with_oai(limit=args.limit)

    output_path = Path(args.output).expanduser().resolve()
    generator.save_csv(csv_data, str(output_path))


if __name__ == "__main__":  # pragma: no cover
    main()
