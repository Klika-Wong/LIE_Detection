# Multimodal Lie Detection Research

This repository contains a partial research implementation for exploring video, audio, and text features in courtroom-trial data. It is a research prototype, not a reliable lie detector. Model outputs must not be treated as evidence about an individual's truthfulness.

## Data manifest

The training entry point expects `samples.json` in the working directory. It must contain a JSON list of records with:

- `text_json_path`: path to a UTF-8 JSON file containing a list of objects with a `text` field.
- `audio_path`: path to an audio file readable by torchaudio.
- `label`: integer class label (0 or 1).
- `frame_images`: optional list of image paths for video frames.

The dataset and paths are private inputs; keep the manifest and source data out of the repository.

## Environment

Install packages from `requirements.txt`. Install a mutually compatible PyTorch, TorchVision, and TorchAudio build for your operating system and accelerator. The script downloads pretrained BERT and ResNet weights when first run and uses Optuna for hyperparameter search.

Run training from this directory with:

```bash
python train.py
```

Training writes checkpoints and a training-curve image under `checkpoints/`. Check the class distribution and evaluate on an independent dataset before interpreting metrics.
