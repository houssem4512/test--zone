import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, min_hand_conf=0.5, min_track_conf=0.5, max_num_hands=1):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_hand_conf,
            min_tracking_confidence=min_track_conf
        )

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        hand_landmarks = None
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

        return hand_landmarks

    def landmarks_to_array(self, hand_landmarks):
        """
        Returns (21, 3) in normalized mediapipe coords (x,y,z).
        """
        lm = []
        for p in hand_landmarks.landmark:
            lm.append([p.x, p.y, p.z])
        return np.array(lm, dtype=np.float32)

    def draw(self, frame_bgr, hand_landmarks):
        if hand_landmarks is None:
            return frame_bgr
        self.mp_draw.draw_landmarks(
            frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )
        return frame_bgr