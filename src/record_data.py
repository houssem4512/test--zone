import os
import json
import cv2
import numpy as np
import argparse

from hand_tracker import HandTracker
from preprocess import normalize_landmarks, resample_sequence
from segmentation import StrokeSegmenter

def load_classes(classes_txt="classes.txt"):
    with open(classes_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Character label to record (must exist in classes.txt)")
    parser.add_argument("--count", type=int, default=60, help="How many samples to record")
    args = parser.parse_args()

    cfg = json.load(open("config.json", "r", encoding="utf-8"))
    seq_len = cfg["sequence_length"]

    classes = load_classes("classes.txt")
    if args.label not in classes:
        raise ValueError(f"Label {args.label} not in classes.txt")

    device_root = os.path.join("data", "samples", args.label)
    os.makedirs(device_root, exist_ok=True)

    tracker = HandTracker(
        min_hand_conf=cfg["min_hand_conf"],
        min_track_conf=cfg["min_track_conf"]
    )

    segmenter = StrokeSegmenter(
        start_vel_thresh=cfg["start_vel_thresh"],
        end_vel_thresh=cfg["end_vel_thresh"],
        start_consecutive=cfg["start_consecutive"],
        end_consecutive=cfg["end_consecutive"],
        max_stroke_frames=cfg["max_stroke_frames"],
        sequence_length=cfg["sequence_length"],
        space_gap_frames=cfg["space_gap_frames"]
    )

    cap = cv2.VideoCapture(cfg["camera_index"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    saved = 0
    sample_idx = len([f for f in os.listdir(device_root) if f.endswith(".npy")])

    print("Recording... Press ESC to quit.")
    print(f"Target label: {args.label}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        hand_landmarks = tracker.process(frame)
        frame = tracker.draw(frame, hand_landmarks)

        event, payload = segmenter.update(
            tracker.landmarks_to_array(hand_landmarks) if hand_landmarks else None
        )

        if event == "stroke_end":
            seq_landmarks = payload["seq_landmarks"]  # (T,21,3)
            # convert each frame to normalized flattened vector
            feats = []
            for t in range(seq_landmarks.shape[0]):
                feats.append(normalize_landmarks(seq_landmarks[t]))
            feats = np.stack(feats, axis=0)  # (T,63)

            feats_rs = resample_sequence(feats, seq_len)  # (seq_len,63)

            out_path = os.path.join(device_root, f"sample_{sample_idx:05d}.npy")
            np.save(out_path, feats_rs)
            sample_idx += 1
            saved += 1
            print(f"Saved {saved}/{args.count}: {out_path}")

        # UI
        cv2.rectangle(frame, (20, 20), (500, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Label: {args.label}", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Saved: {saved}/{args.count}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

        cv2.imshow("AirWrite Recorder (stroke auto-saves)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if saved >= args.count:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()