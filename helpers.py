import numpy as np
import cv2

class PreprocessFrame:
    def __init__(self, width=16, height=16, grayscale=True):
        self.width = width
        self.height = height
        self.grayscale = grayscale

    def preprocess(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame / 255.0
        frame = (frame > 0.5).astype(np.float32)  # Binary 64x64x1
        return frame.flatten()  # Shape: (4096,)

    def reset(self, frame):
        return self.preprocess(frame)

    def step(self, frame):
        return self.preprocess(frame)
