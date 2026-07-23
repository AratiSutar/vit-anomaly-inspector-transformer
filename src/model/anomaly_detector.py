import torch
import numpy as np
from src.model.vit import ViT


class MemoryBank:
    def __init__(self, num_patches=64, d_model=128, eps=0.01):
        self.num_patches = num_patches
        self.d_model = d_model
        self.eps = eps
        self.embeddings = [[] for _ in range(num_patches)]
        self.means = None
        self.inv_covs = None
    
    def collect(self, patch_embeddings):
        # patch_embeddings shape: (B, 64, 128)
        B, num_patches, d_model = patch_embeddings.shape
        for p in range(num_patches):
            for b in range(B):
                self.embeddings[p].append(
                    patch_embeddings[b, p, :].detach().cpu().numpy()
                )

    def fit(self):
        self.means = []
        self.inv_covs = []

        for p in range(self.num_patches):
            # stack all embeddings for patch position p
            # shape: (N, 128)
            X = np.stack(self.embeddings[p], axis=0)

            # compute mean
            mean = np.mean(X, axis=0)

            # compute covariance
            cov = np.cov(X, rowvar=False)

            # regularization — add eps to diagonal
            cov += self.eps * np.eye(self.d_model)

            # invert
            inv_cov = np.linalg.inv(cov)

            self.means.append(mean)
            self.inv_covs.append(inv_cov)

    def score(self, patch_embeddings):
        # patch_embeddings shape: (B, 64, 128)
        B, num_patches, d_model = patch_embeddings.shape
        patch_embeddings = patch_embeddings.detach().cpu().numpy()
        anomaly_maps = []

        for b in range(B):
            scores = []
            for p in range(num_patches):
                x = patch_embeddings[b, p, :]
                mu = self.means[p]
                inv_cov = self.inv_covs[p]

                # deviation from normal center
                diff = x - mu

                # mahalanobis distance
                score = np.sqrt(diff @ inv_cov @ diff)
                scores.append(score)

            # reshape 64 scores into 8x8 grid
            anomaly_map = np.array(scores).reshape(8, 8)
            anomaly_maps.append(anomaly_map)

        return np.stack(anomaly_maps, axis=0)
    
    class AnomalyDetector:
    def __init__(self, vit_model, device='cpu'):
        self.vit = vit_model
        self.vit.eval()
        self.device = device
        self.memory_bank = MemoryBank()

    def build_memory_bank(self, dataloader):
        print("Building memory bank from normal training images...")
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                patch_embeddings = self.vit(images)
                self.memory_bank.collect(patch_embeddings)
        self.memory_bank.fit()
        print("Memory bank ready.")

    def predict(self, images):
        self.vit.eval()
        with torch.no_grad():
            images = images.to(self.device)
            patch_embeddings = self.vit(images)
        anomaly_maps = self.memory_bank.score(patch_embeddings)
        scores = anomaly_maps.max(axis=(1, 2))
        return scores, anomaly_maps
