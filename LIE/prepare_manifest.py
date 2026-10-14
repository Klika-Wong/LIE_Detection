#!/usr/bin/env python3
"""Prepare a local samples.json manifest and video frames for LIE/train.py.

The source dataset is read-only. Generated frames and the manifest are written
outside the dataset directory. Speaker tags in dataset segment metadata are
intentionally ignored because their identity scope is not known.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image


VIDEO_EXTENSIONS = {".mp4"}
AUDIO_EXTENSIONS = {".wav"}
LABEL_FOLDERS = (("Deceptive", 1), ("Truthful", 0))


def fail(message):
    raise ValueError(message)


def ensure_outside_dataset(path, dataset_root, label):
    resolved_path = path.resolve()
    resolved_root = dataset_root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return resolved_path
    fail(f"{label} must be outside the source dataset directory: {resolved_path}")


def index_files(folder, extensions, description):
    if not folder.is_dir():
        fail(f"Missing {description} directory: {folder}")
    indexed = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.stem
            if key in indexed:
                fail(f"Duplicate {description} clip ID: {key}")
            indexed[key] = path.resolve()
    return indexed


def read_group_mapping(csv_path):
    groups = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"clip_id", "group_id"}.issubset(reader.fieldnames):
            fail("Groups CSV must have columns named clip_id and group_id.")
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            group_id = (row.get("group_id") or "").strip()
            if not clip_id or not group_id:
                fail("Groups CSV contains an empty clip_id or group_id.")
            if clip_id in groups:
                fail(f"Groups CSV contains duplicate clip_id: {clip_id}")
            groups[clip_id] = group_id
    return groups


def read_transcript(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Cannot read transcript JSON {path.name}: {exc}")
    if not isinstance(payload, list) or not payload:
        fail(f"Transcript JSON must be a non-empty list: {path.name}")
    if any(not isinstance(item, dict) or not isinstance(item.get("text"), str)
           or not item["text"].strip() for item in payload):
        fail(f"Each transcript segment needs non-empty text: {path.name}")


def extract_frames(video_path, output_dir, frame_count):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"Could not open video: {video_path.name}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        fail(f"Video has no readable frames: {video_path.name}")

    if frame_count == 1:
        indices = [total_frames // 2]
    else:
        indices = [round(i * (total_frames - 1) / (frame_count - 1))
                   for i in range(frame_count)]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        for index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                fail(f"Could not read frame {frame_index} from {video_path.name}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_path = output_dir / f"frame_{index:03d}.png"
            image.save(frame_path)
            saved.append(str(frame_path.resolve()))
    finally:
        capture.release()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Build a local training manifest from the private LIE dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Dataset directory containing Clips/Deceptive and Clips/Truthful.")
    parser.add_argument("--transcripts-dir", type=Path, required=True,
                        help="Directory of transcript JSON files named <clip_id>.json.")
    parser.add_argument("--groups-csv", type=Path, required=True,
                        help="CSV with clip_id,group_id columns and verified subject IDs.")
    parser.add_argument("--output", type=Path, default=Path("samples.json"))
    parser.add_argument("--frame-dir", type=Path, default=Path("prepared_frames"))
    parser.add_argument("--frames-per-video", type=int, default=16)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if args.frames_per_video < 1:
        fail("--frames-per-video must be at least 1.")
    if not dataset_root.is_dir():
        fail(f"Dataset directory does not exist: {dataset_root}")
    if not args.transcripts_dir.is_dir():
        fail(f"Transcript directory does not exist: {args.transcripts_dir}")
    if not args.groups_csv.is_file():
        fail(f"Groups CSV does not exist: {args.groups_csv}")

    output_path = ensure_outside_dataset(args.output, dataset_root, "Manifest")
    frame_root = ensure_outside_dataset(args.frame_dir, dataset_root, "Frame output")
    if output_path == frame_root or frame_root in output_path.parents:
        fail("Manifest output must not be placed inside the generated frame directory.")

    transcript_paths = index_files(args.transcripts_dir, {".json"}, "transcript")
    groups = read_group_mapping(args.groups_csv)
    samples = []
    seen_ids = set()

    for folder_name, label in LABEL_FOLDERS:
        clip_dir = dataset_root / "Clips" / folder_name
        videos = index_files(clip_dir, VIDEO_EXTENSIONS, f"{folder_name} video")
        audios = index_files(clip_dir, AUDIO_EXTENSIONS, f"{folder_name} audio")
        if not videos or not audios:
            fail(f"No MP4 videos or WAV audio found in {clip_dir}")

        if set(videos) != set(audios):
            missing_audio = sorted(set(videos) - set(audios))
            missing_video = sorted(set(audios) - set(videos))
            fail(
                f"Unpaired files in {folder_name}: "
                f"{len(missing_audio)} videos without WAV and "
                f"{len(missing_video)} WAV files without video."
            )

        for clip_id in sorted(videos):
            if clip_id in seen_ids:
                fail(f"Clip ID appears in both label folders: {clip_id}")
            seen_ids.add(clip_id)
            transcript_path = transcript_paths.get(clip_id)
            if transcript_path is None:
                fail(f"Missing transcript JSON for clip ID: {clip_id}")
            if clip_id not in groups:
                fail(f"Missing verified subject group_id for clip ID: {clip_id}")
            read_transcript(transcript_path)
            frame_paths = extract_frames(
                videos[clip_id], frame_root / clip_id, args.frames_per_video
            )
            samples.append({
                "text_json_path": str(transcript_path),
                "audio_path": str(audios[clip_id]),
                "label": label,
                "group_id": groups[clip_id],
                "frame_images": frame_paths,
            })

    extra_group_ids = set(groups) - seen_ids
    if extra_group_ids:
        fail(f"Groups CSV contains {len(extra_group_ids)} clip IDs absent from the dataset.")
    extra_transcripts = set(transcript_paths) - seen_ids
    if extra_transcripts:
        fail(f"Transcript folder contains {len(extra_transcripts)} clip IDs absent from the dataset.")
    if not samples:
        fail("No paired samples were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_resolved = output_path.resolve()
    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=output_path_resolved.parent,
        delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(samples, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, output_path_resolved)
    print(f"Prepared {len(samples)} samples at {output_path_resolved}")
    print(f"Extracted frames under {frame_root}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
