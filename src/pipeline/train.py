import torch
from torch.utils.data import DataLoader
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.vit import ViT
from src.model.anomaly_detector import AnomalyDetector
from src.pipeline.dataset import MVTecDataset


def train(category='bottle', data_root='data/mvtec', batch_size=16, save_path='outputs'):
    device = torch.device('cpu')

    print(f"Loading MVTec AD - category: {category}")
    dataset = MVTecDataset(root_dir=data_root, category=category, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Initializing ViT...")
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
    detector.build_memory_bank(dataloader)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{category}_memory_bank.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump({
            'means': detector.memory_bank.means,
            'inv_covs': detector.memory_bank.inv_covs
        }, f)

    print(f"Memory bank saved to {save_file}")
    return detector


if __name__ == '__main__':
    train()
