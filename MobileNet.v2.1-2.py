import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from thop import profile
import time
from tqdm import tqdm

# ==== CONFIG ====
DATA_DIR = r"E:\project\datasets\ds_5"  # path to dataset
LABEL_FILE = None  # not needed if ImageFolder used, else CSV if same as training
MODEL_PATH = r"E:\project\models\checkpoints_mobilenet_v2.2.1\mobilenetv2_epoch_10_v2.2.1.pth"
IMAGE_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1

# ==== CLAHE Transform ====
class CLAHE:
    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return transforms.ToPILImage()(final_img)

# ==== Transform (same as training but without heavy random aug for test) ====
test_transform = transforms.Compose([
    CLAHE(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Dataset (matching your training DeepfakeDataset) ====
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        for label_name, label in [('fake', 0), ('ffhq', 1)]:
            folder = os.path.join(root_dir, label_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append(os.path.join(folder, fname))
                    self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        min_dim = min(h, w)
        startx = w // 2 - (min_dim // 2)
        starty = h // 2 - (min_dim // 2)
        image = image[starty:starty + min_dim, startx:startx + min_dim]
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ==== DataLoader ====
test_dataset = DeepfakeDataset(os.path.join(DATA_DIR, "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Model (exact v2.2 structure) ====
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.last_channel, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.3),
    nn.Linear(512, 1)
)

# ==== Load weights ====
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model = model.to(DEVICE)
model.eval()

# ==== FLOPs & Params ====
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
print(f"Params: {params/1e6:.2f} M")

# ==== Inference Time ====
times = []
with torch.no_grad():
    for idx, (img, _) in enumerate(test_loader):
        if idx >= 50:  # measure on first 50
            break
        img = img.to(DEVICE)
        start = time.time()
        _ = model(img)
        end = time.time()
        times.append(end - start)
avg_time = np.mean(times)
print(f"Avg inference time per image: {avg_time*1000:.2f} ms")

# ==== ROC-AUC ====
sigmoid = nn.Sigmoid()
y_true, y_scores = [], []
with torch.no_grad():
    for img, label in tqdm(test_loader, desc="Evaluating"):
        img = img.to(DEVICE)
        output = model(img)
        prob = sigmoid(output).cpu().numpy().flatten()
        y_true.extend([label])
        y_scores.extend(prob)

auc = roc_auc_score(y_true, y_scores)
print(f"AUC Score: {auc:.4f}")
