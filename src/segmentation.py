import numpy as np

class StrokeSegmenter:
    def __init__(
        self,
        start_vel_thresh=0.015,
        end_vel_thresh=0.008,
        start_consecutive=3,
        end_consecutive=8,
        max_stroke_frames=120,
        sequence_length=64,
        space_gap_frames=35
    ):
        self.start_vel_thresh = start_vel_thresh
        self.end_vel_thresh = end_vel_thresh
        self.start_consecutive = start_consecutive
        self.end_consecutive = end_consecutive
        self.max_stroke_frames = max_stroke_frames
        self.sequence_length = sequence_length
        self.space_gap_frames = space_gap_frames

        self.reset()

    def reset(self):
        self.in_stroke = False
        self.buffer = []
        self.prev_wrist = None

        self.start_count = 0
        self.end_count = 0

        self.frames_since_end = 10**9
        self.last_stroke_ended = False

    def update(self, lm21_xyz: np.ndarray):
        """
        lm21_xyz: (21,3) or None when no hand.
        Returns:
          (event, payload)
          event in {"none","stroke_end","space"}
          payload for stroke_end: dict with "seq" (target_len,F)
        """
        if lm21_xyz is None:
            # no hand detected
            self.frames_since_end += 1
            return "none", None

        # wrist movement magnitude (normalized coords)
        wrist = lm21_xyz[0].astype(np.float32)
        if self.prev_wrist is None:
            vel = 0.0
        else:
            vel = float(np.linalg.norm(wrist - self.prev_wrist))
        self.prev_wrist = wrist

        self.frames_since_end += 1

        if not self.in_stroke:
            if vel > self.start_vel_thresh:
                self.start_count += 1
            else:
                self.start_count = 0

            if self.start_count >= self.start_consecutive:
                self.in_stroke = True
                self.buffer = []
                self.end_count = 0
                self.start_count = 0

            return "none", None

        # If in stroke: collect
        self.buffer.append(lm21_xyz)

        # end detection
        if vel < self.end_vel_thresh:
            self.end_count += 1
        else:
            self.end_count = 0

        if self.end_count >= self.end_consecutive or len(self.buffer) >= self.max_stroke_frames:
            self.in_stroke = False
            self.frames_since_end = 0
            seq = np.stack(self.buffer, axis=0)  # (T,21,3)
            self.buffer = []
            self.end_count = 0

            # return normalized + resampled later in app/dataset pipeline
            return "stroke_end", {"seq_landmarks": seq}

        return "none", None

    def should_insert_space(self):
        return self.frames_since_end >= self.space_gap_frames