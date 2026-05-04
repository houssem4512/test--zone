import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task"):

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ✅ FIX: correct MediaPipe Tasks input
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        hand_landmarks = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

        return hand_landmarks

    def landmarks_to_array(self, hand_landmarks):
        """
        Returns (21, 3) normalized coordinates (x, y, z)
        """
        if hand_landmarks is None:
            return None

        lm = []
        for p in hand_landmarks:
            lm.append([p.x, p.y, p.z])

        return np.array(lm, dtype=np.float32)

    def draw(self, frame_bgr, hand_landmarks):
        # Optional: simple debug drawing (not full skeleton)
        if hand_landmarks is None:
            return frame_bgr

        h, w, _ = frame_bgr.shape

        for p in hand_landmarks:
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1)

        return frame_bgr