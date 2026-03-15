from typing import Tuple
import numpy as np
from sklearn.utils import resample

class RejectInference:
    def __init__(self, augmentation_factor: float = 1.5, parceling_rate: float = 0.3):
        self.augmentation_factor = augmentation_factor
        self.parceling_rate = parceling_rate

    def augment_rejects(self, X_rejects: np.ndarray, y_rejects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment rejected samples with synthetic labels."""
        n_samples = int(len(X_rejects) * self.augmentation_factor)
        X_aug, y_aug = resample(X_rejects, y_rejects, n_samples=n_samples, random_state=42)
        return X_aug, y_aug

    def parceling(self, X_rejects: np.ndarray) -> np.ndarray:
        """Assign pseudo-labels to rejects via parceling."""
        n_parcel = int(len(X_rejects) * self.parceling_rate)
        pseudo_labels = np.zeros(len(X_rejects))
        pseudo_labels[:n_parcel] = 1  # Assume some are good
        np.random.shuffle(pseudo_labels)
        return pseudo_labels
