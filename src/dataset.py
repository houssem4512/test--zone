import os
import glob
import numpy as np
from torch.utils.data import Dataset

class NpySequenceDataset(Dataset):
    def __init__(self, samples_root, classes):
        self.samples = []
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            folder = os.path.join(samples_root, c)
            if not os.path.isdir(folder):
                continue
            for path in glob.glob(os.path.join(folder, "*.npy")):
                self.samples.append((path, self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {samples_root}. Record data first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)  # (T, F) float32
        return x, label