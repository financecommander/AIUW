from typing import Tuple
import numpy as np
from sklearn.utils import resample

class RejectInference:
    def __init__(self, augmentation_factor: float = 1.5, parceling_ratio: float = 0.3):
        self.augmentation_factor = augmentation_factor
        self.parceling_ratio = parceling_ratio

    def augment_rejects(self, X_rejects: np.ndarray, y_rejects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment rejected samples with synthetic labels."""
        n_samples = int(len(X_rejects) * self.augmentation_factor)
        X_aug, y_aug = resample(X_rejects, y_rejects, n_samples=n_samples, replace=True)
        return X_aug, y_aug

    def parceling(self, X_rejects: np.ndarray) -> np.ndarray:
        """Parcel rejected applications into risk bands (placeholder logic)."""
        n_samples = len(X_rejects)
        n_parceled = int(n_samples * self.parceling_ratio)
        return np.random.choice(n_samples, n_parceled, replace=False)

    def apply(self, X_accepted: np.ndarray, y_accepted: np.ndarray, 
              X_rejects: np.ndarray, y_rejects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_aug, y_aug = self.augment_rejects(X_rejects, y_rejects)
        X_combined = np.vstack([X_accepted, X_aug])
        y_combined = np.hstack([y_accepted, y_aug])
        return X_combined, y_combined
