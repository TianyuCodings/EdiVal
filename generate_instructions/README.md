# EdiVal Instruction Pipeline

This directory contains the end-to-end workflow for creating instruction datasets used by EdiVal. There are three main stages:

1. **Object extraction:** use the OpenAI API to list every salient object in each input image.
2. **Grounding filter:** verify each candidate object with Grounding DINO and keep only visually grounded items.
3. **Instruction generation:** build multi-turn editing instructions with the OpenAI API.

All scripts share the same directory layout and cache their outputs inside `generate_instructions/`.

## Prerequisites

- Activate the `edival` environment (`conda activate edival`) and ensure Grounding DINO weights are downloaded (`GroundingDINO/weights/groundingdino_swint_ogc.pth`).
- Set `OPENAI_API_KEY` before running any OpenAI-powered steps (object extraction or instruction generation). Optionally configure `OPENAI_API_BASE` if you use a custom endpoint.
- Place the resized input images (e.g. `input_images_resize_512/` or `.zip`) at the repository root. Update paths with CLI flags if you use a different location.
- Candidate vocabulary files live in `generate_instructions/candidate_pools/`. Update or regenerate them if you need custom object/color/material lists.

## 1. Extract Objects (`oai_all_objects.py`)

Generates rich JSON metadata for each image by calling the OpenAI API.

```bash
export OPENAI_API_KEY="..."                     # or use a .env file
python generate_instructions/oai_all_objects.py \
  --input-dir input_images_resize_512 \
  --output-dir generate_instructions/oai_all_objects
```

Outputs: one `<index>_input_raw.json` per image under `generate_instructions/oai_all_objects/`.

## 2. Ground and Filter Objects (`grounding_filter.py`)

Runs Grounding DINO over the raw JSON to keep only objects that can be localized in the image and adds grounding metadata.

```bash
python generate_instructions/grounding_filter.py \
  --input-dir generate_instructions/oai_all_objects \
  --output-dir generate_instructions/grounding_all_objects \
  --image-dir input_images_resize_512 \
  --num-gpus 2 \
  --box-threshold 0.35 \
  --text-threshold 0.35
```

Adjust GPU count, thresholds, or directories as needed. If no GPUs are available, pass `--cpu` to run on the CPU (much slower).

Outputs: filtered JSONs under `generate_instructions/grounding_all_objects/`.

## 3. Generate Instructions (`oai_instruction_generator.py`)

Produces the final CSV file consumed by the rest of the pipeline using the OpenAI API (the same key from step 1).

```bash
export OPENAI_API_KEY="..."                     # ensure key is available
python generate_instructions/oai_instruction_generator.py \
  --grounding-dir generate_instructions/grounding_all_objects \
  --input-images input_images_resize_512 \
  --output oai_instruction_generation_output.csv \
  --seed 42
```

Use `--limit N` to process a subset of images for quick verification.

## Tips

- All CLI flags accept absolute or relative paths; directories are created automatically.
- Logs are verbose—redirect to files (e.g. `... > logs/step1.log`) for long runs.
- Once the CSV is generated, downstream generation and evaluation scripts should point to `oai_instruction_generation_output.csv`.
- To refresh the candidate pools, optionally run `generate_instructions/candidate_pools/generate_objects_txt.py` against a folder of filtered JSONs; it will rebuild the `*.txt` files in that directory.
