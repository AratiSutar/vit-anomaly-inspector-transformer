import torch
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.vit import ViT
from src.model.anomaly_detector import AnomalyDetector
from src.pipeline.dataset import MVTecDataset


def evaluate(category='bottle', data_root='data/mvtec',
             batch_size=16, save_path='outputs'):

    device = torch.device('cpu')

    # load saved memory bank
    save_file = os.path.join(save_path, f'{category}_memory_bank.pkl')
    with open(save_file, 'rb') as f:
        memory_bank_data = pickle.load(f)

    # rebuild vit and detector
    vit = ViT(
        image_size=64,
        patch_size=8,
        in_channels=3,
        d_model=128,
        num_heads=4,
        num_layers=6
    )
    vit.to(device)

    detector = AnomalyDetector(vit_model=vit, device=device)
    detector.memory_bank.means = memory_bank_data['means']
    detector.memory_bank.inv_covs = memory_bank_data['inv_covs']

    # load test dataset
    test_dataset = MVTecDataset(root_dir=data_root,
                                category=category, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    all_scores = []
    all_labels = []
    all_maps = []

    for images, labels in test_loader:
        scores, anomaly_maps = detector.predict(images)
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        all_maps.extend(anomaly_maps)

    # compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    print(f"AUROC for {category}: {auroc:.4f}")

    # save anomaly maps visualization
    visualize(all_maps, all_labels, all_scores, category, save_path)

    return auroc


def visualize(maps, labels, scores, category, save_path):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Anomaly Maps - {category}')

    defect_indices = [i for i, l in enumerate(labels) if l == 1][:5]
    normal_indices = [i for i, l in enumerate(labels) if l == 0][:5]

    for col, (d_idx, n_idx) in enumerate(zip(defect_indices, normal_indices)):
        axes[0, col].imshow(maps[d_idx], cmap='hot')
        axes[0, col].set_title(f'Defect\nscore:{scores[d_idx]:.2f}')
        axes[0, col].axis('off')

        axes[1, col].imshow(maps[n_idx], cmap='hot')
        axes[1, col].set_title(f'Normal\nscore:{scores[n_idx]:.2f}')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{category}_anomaly_maps.png'))
    print(f"Visualization saved.")


if __name__ == '__main__':
    evaluate()
