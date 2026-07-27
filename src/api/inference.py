import torch
import pickle
import os
import sys
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.vit import ViT
from src.model.anomaly_detector import AnomalyDetector


class InferenceEngine:
    def __init__(self, category='bottle', save_path='outputs'):
        self.device = torch.device('cpu')
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # load memory bank
        memory_bank_file = os.path.join(save_path, f'{category}_memory_bank.pkl')
        with open(memory_bank_file, 'rb') as f:
            memory_bank_data = pickle.load(f)

        # load threshold
        threshold_file = os.path.join(save_path, f'{category}_threshold.pkl')
        with open(threshold_file, 'rb') as f:
            threshold_data = pickle.load(f)

        self.threshold = threshold_data['threshold']

        # rebuild vit and detector
        vit = ViT(
            image_size=64,
            patch_size=8,
            in_channels=3,
            d_model=128,
            num_heads=4,
            num_layers=6
        )
        vit.to(self.device)

        self.detector = AnomalyDetector(vit_model=vit, device=self.device)
        self.detector.memory_bank.means = memory_bank_data['means']
        self.detector.memory_bank.inv_covs = memory_bank_data['inv_covs']

    def predict_from_bytes(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)

        scores, anomaly_maps = self.detector.predict(tensor)
        score = float(scores[0])

        result = {
            "anomaly_score": round(score, 4),
            "threshold": round(float(self.threshold), 4),
            "prediction": "DEFECTIVE" if score > self.threshold else "NORMAL",
            "anomaly_map_shape": list(anomaly_maps[0].shape)
        }
        return result
