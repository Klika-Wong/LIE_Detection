# Multimodal Lie Detection Research

This repository contains a partial research implementation for exploring video, audio, and text features in courtroom-trial data. It is a research prototype, not a reliable lie detector. Model outputs must not be treated as evidence about an individual's truthfulness.

## Preparing local data

The private dataset and its media must stay outside this repository. The preparation script reads the dataset and writes only a local manifest and extracted frames. It expects the dataset layout to contain `Clips/Deceptive/` and `Clips/Truthful/`, with a matching `.mp4` video and `.wav` audio file for each clip stem.

The supplied segment JSON files contain timing/speaker metadata but no transcript text. Provide transcript JSON files separately, one per clip, named `<clip_id>.json`. Each must be a non-empty list of objects with non-empty `text` fields, for example:

```json
[
  {"text": "Transcript segment one."},
  {"text": "Transcript segment two."}
]
```

Create a CSV with the header `clip_id,group_id` and one row for every clip. `group_id` must identify the verified real subject so all clips from that person remain in one validation split. The scope of the dataset's `speaker` tags is unknown, so do not copy those tags into `group_id`; use verified dataset metadata or another reliable mapping. The training script stops if either split lacks one of the two labels.

Install packages from `requirements.txt`, then run from the repository root:

```bash
python LIE/prepare_manifest.py \
  --dataset-root "/path/to/Lie_Dec_datasets" \
  --transcripts-dir "/path/to/transcripts" \
  --groups-csv "/path/to/groups.csv" \
  --output samples.json \
  --frame-dir prepared_frames
python LIE/train.py
```

The script samples 16 frames from each video by default. The number can be changed with `--frames-per-video`. Paths in `samples.json` are absolute. The manifest, generated frames, and source data should remain local and must not be committed.

## Environment and evaluation

Install a mutually compatible PyTorch, TorchVision, and TorchAudio build for your operating system and accelerator. The training script downloads pretrained BERT and ResNet weights when first run and uses Optuna for hyperparameter search.

Training writes checkpoints and a training-curve image under `checkpoints/`. Check the class distribution and evaluate on an independent dataset before interpreting metrics. This repository does not create an independent test set, so validation metrics alone are not final performance estimates.
