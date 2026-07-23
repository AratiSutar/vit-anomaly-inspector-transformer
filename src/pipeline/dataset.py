#we load MVTec AD dataset and prepare it for the ViT
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', image_size=64):
        self.mode = mode
        self.image_size = image_size
        self.samples = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if mode == 'train':
            good_dir = os.path.join(root_dir, category, 'train', 'good')
            for fname in os.listdir(good_dir):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    self.samples.append(os.path.join(good_dir, fname))
                    self.labels.append(0)

        elif mode == 'test':
            test_dir = os.path.join(root_dir, category, 'test')
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                label = 0 if defect_type == 'good' else 1
                for fname in os.listdir(defect_dir):
                    if fname.endswith('.png') or fname.endswith('.jpg'):
                        self.samples.append(os.path.join(defect_dir, fname))
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]
