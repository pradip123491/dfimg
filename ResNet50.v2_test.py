import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm import create_model
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image

# ─────────────────────────────────────
# 🔧 Config
# ─────────────────────────────────────
torch.cuda.set_device(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

BASE_DIR = '/home/shavak/YC/project/datasets/ds_5'
CHECKPOINT_DIR = '/home/shavak/YC/project/models/checkpoints_resnet50_v2.0'
EPOCH_TO_TEST = 16
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'resnet50_epoch_{EPOCH_TO_TEST}_v2.0.pth')

IMAGE_SIZE = 512
BATCH_SIZE = 16

# ─────────────────────────────────────
# 🎨 CLAHE
# ─────────────────────────────────────
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

# ─────────────────────────────────────
# 🔄 Transform
# ─────────────────────────────────────
test_transform = transforms.Compose([
    CLAHE(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────
# 📁 Load Test Data
# ─────────────────────────────────────
test_dir = os.path.join(BASE_DIR, 'test')
test_ds = ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_ds.classes

# ─────────────────────────────────────
# 🧠 Load ResNet50 (Timm)
# ─────────────────────────────────────
model = create_model('resnet50', pretrained=False, num_classes=1)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(model.get_classifier().in_features, 1)
)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model_state'])
model = model.to(DEVICE)
model.eval()

print(f"\n🔎 Testing ResNet50 @ Epoch {EPOCH_TO_TEST}...")

# ─────────────────────────────────────
# 🧪 Inference
# ─────────────────────────────────────
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="🔍 Testing", ncols=100):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ─────────────────────────────────────
# 📊 Evaluation
# ─────────────────────────────────────
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names)

print(f"\n✅ Test Accuracy @ Epoch {EPOCH_TO_TEST}: {acc:.4f}")
print("\n📋 Classification Report:\n", report)

# ─────────────────────────────────────
# 🖼️ Confusion Matrix Plot
# ─────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Epoch {EPOCH_TO_TEST})')
plt.tight_layout()

plot_path = os.path.join(CHECKPOINT_DIR, f'confusion_matrix_epoch_{EPOCH_TO_TEST}.png')
plt.savefig(plot_path)
print(f"📸 Confusion matrix saved to: {plot_path}")
