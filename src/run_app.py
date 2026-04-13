import json
import os
import cv2
import numpy as np
import torch

from hand_tracker import HandTracker
from preprocess import normalize_landmarks, resample_sequence
from segmentation import StrokeSegmenter
from model import GRUClassifier

def load_classes(classes_txt="classes.txt"):
    with open(classes_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    cfg = json.load(open("config.json", "r", encoding="utf-8"))
    classes = load_classes("classes.txt")

    ckpt_path = os.path.join("models", "gru_airwrite_best.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Model not found: {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes_ckpt = ckpt["classes"]
    seq_len = ckpt["seq_len"]
    F = ckpt["F"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRUClassifier(
        input_size=F,
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=len(classes_ckpt)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

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
        sequence_length=seq_len,
        space_gap_frames=cfg["space_gap_frames"]
    )

    cap = cv2.VideoCapture(cfg["camera_index"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    typed_text = ""
    last_pred = ""
    last_conf = 0.0

    print("Running real-time app.")
    print("Controls: ESC quit | c clear | s add space | b backspace")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        hand_landmarks = tracker.process(frame)
        frame = tracker.draw(frame, hand_landmarks)

        lm21 = tracker.landmarks_to_array(hand_landmarks) if hand_landmarks else None
        event, payload = segmenter.update(lm21)

        if event == "stroke_end":
            seq_landmarks = payload["seq_landmarks"]  # (T,21,3)

            feats = []
            for t in range(seq_landmarks.shape[0]):
                feats.append(normalize_landmarks(seq_landmarks[t]))
            feats = np.stack(feats, axis=0)  # (T,63)

            feats_rs = resample_sequence(feats, seq_len)  # (seq_len,63)

            x = torch.tensor(feats_rs[None, :, :], dtype=torch.float32).to(device)  # (1,T,F)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)

            conf = float(conf.item())
            idx = int(idx.item())
            pred_char = classes_ckpt[idx]
            last_pred = pred_char
            last_conf = conf

            if conf >= cfg["inference"]["min_conf"]:
                # auto-space: if gap was large since last stroke end
                if segmenter.should_insert_space() and (len(typed_text) > 0) and (typed_text[-1] != " "):
                    typed_text += " "

                typed_text += pred_char

        # --- Draw “phrase box” on frame ---
        h, w = frame.shape[:2]
        box_h = 110
        cv2.rectangle(frame, (0, h - box_h), (w, h), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, h - box_h + 10), (w - 10, h - 10), (40, 40, 40), 2)

        cv2.putText(frame, "AIR WRITE:", (20, h - box_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, typed_text[-40:], (20, h - box_h + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220,255,220), 2)

        cv2.putText(frame, f"Last: {last_pred} ({last_conf:.2f})",
                    (w - 380, h - box_h + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,255), 1)

        cv2.imshow("AirWrite (phrase box) - press ESC to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            typed_text = ""
        elif key == ord('s'):
            if len(typed_text) == 0 or typed_text[-1] != " ":
                typed_text += " "
        elif key == ord('b'):
            typed_text = typed_text[:-1]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()