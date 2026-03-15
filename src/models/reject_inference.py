from typing import Tuple
import numpy as np
from sklearn.utils import resample

class RejectInference:
    def __init__(self, method: str = 'augmentation', parcel_size: int = 1000):
        self.method = method
        self.parcel_size = parcel_size

    def augment_rejects(self, X_accepted: np.ndarray, y_accepted: np.ndarray, 
                       X_rejected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == 'augmentation':
            n_rejects = len(X_rejected)
            sampled_rejects = resample(X_rejected, n_samples=n_rejects // 2)
            pseudo_labels = np.zeros(len(sampled_rejects))  # Assume default for rejects
            X_combined = np.vstack([X_accepted, sampled_rejects])
            y_combined = np.hstack([y_accepted, pseudo_labels])
            return X_combined, y_combined
        return X_accepted, y_accepted  # Fallback

    def parcel_data(self, X: np.ndarray, y: np.ndarray) -> list:
        parcels = []
        n_batches = len(X) // self.parcel_size
        for i in range(n_batches):
            start_idx = i * self.parcel_size
            end_idx = (i + 1) * self.parcel_size
            parcels.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        return parcels
