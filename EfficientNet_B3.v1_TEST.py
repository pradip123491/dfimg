# EfficientNet B3 - Test Script
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────
torch.cuda.set_device(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

IMAGE_SIZE = 512
BATCH_SIZE = 16
BASE_DIR = '/home/shavak/YC/project/datasets/ds_5/test'
CHECKPOINT_PATH = '/home/shavak/YC/project/models/checkpoints_efficientnet_v1.0/best_efficientnet_v1.0.pth'
LOG_FILE = '/home/shavak/YC/project/models/checkpoints_efficientnet_v1.0/test_log.txt'

# ──────────────────────────────────────────────────
# 🎨 CLAHE Transform
# ──────────────────────────────────────────────────
class CLAHE:
    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)

# ──────────────────────────────────────────────────
# Transform & Dataset
# ──────────────────────────────────────────────────
transform = transforms.Compose([
    CLAHE(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class TestImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(folder_path)))}
        self.classes = list(self.class_to_idx.keys())

        for cls in self.classes:
            cls_folder = os.path.join(folder_path, cls)
            for fname in os.listdir(cls_folder):
                self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label

# ──────────────────────────────────────────────────
# Load Dataset
# ──────────────────────────────────────────────────
test_ds = TestImageFolder(BASE_DIR, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("📂 Classes:", test_ds.classes)
print("📁 Class-to-Index Mapping:", test_ds.class_to_idx)

# ──────────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────────
model = create_model('efficientnet_b3', pretrained=False, num_classes=1)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model_state'])
model = model.to(DEVICE)
model.eval()

# ──────────────────────────────────────────────────
# Test Loop
# ──────────────────────────────────────────────────
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="🔍 Evaluating"):
        imgs = imgs.to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE)
        outputs = model(imgs)
        preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ──────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────
acc = np.mean(np.array(all_preds) == np.array(all_labels))
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=test_ds.classes)

with open(LOG_FILE, 'w') as f:
    f.write(f"✅ Test Accuracy: {acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

print("✅ Test Accuracy:", f"{acc:.4f}")
print("📊 Confusion Matrix:\n", cm)
print("📄 Classification Report:\n", report)
