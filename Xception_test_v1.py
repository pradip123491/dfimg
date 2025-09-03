# import os
# import cv2
# import time
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from datetime import datetime
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# from timm import create_model
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ───────────────────────────────────────
# # Config
# # ───────────────────────────────────────
# DEVICE = torch.device("cuda")  # or "cpu"
# print("🚀 Using:", DEVICE)

# BASE_DIR = 'E:/project/datasets/ds_5'
# IMAGE_SIZE = 512
# BATCH_SIZE = 16
# CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "..", "models", "checkpoints_xception_v1.0", "xception_epoch_12_v1.0.pth")

# # CHECKPOINT_PATH = '/home/shavak/YC/project/models/checkpoints_xception_v1.0/xception_epoch_12_v1.0.pth'

# # ───────────────────────────────────────
# # Safe CLAHE Transform
# # ───────────────────────────────────────
# class CLAHE:
#     def __call__(self, img):
#         try:
#             img_np = np.array(img)
#             lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             merged = cv2.merge((cl, a, b))
#             final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
#             return Image.fromarray(final_img)
#         except Exception as e:
#             print(f"⚠️ CLAHE failed: {e}")
#             return img

# # ───────────────────────────────────────
# # Auto Image Checker (with progress bar)
# # ───────────────────────────────────────
# def validate_images(folder):
#     print(f"🧪 Validating images in: {folder}")
#     corrupted = []
#     all_files = []
#     for root, _, files in os.walk(folder):
#         for f in files:
#             all_files.append(os.path.join(root, f))

#     for path in tqdm(all_files, desc="📷 Checking images", ncols=100):
#         try:
#             img = Image.open(path).convert("RGB")
#             img.verify()
#         except Exception as e:
#             print(f"❌ Corrupt or unreadable: {path} — {e}")
#             corrupted.append(path)

#     print(f"✅ Image check done. Corrupted: {len(corrupted)}\n")
#     return corrupted

# # ───────────────────────────────────────
# # Transforms
# # ───────────────────────────────────────
# transform = transforms.Compose([
#     CLAHE(),
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # Validate test set
# corrupted_files = validate_images(os.path.join(BASE_DIR, 'test')) # test dir

# # Load Dataset (with progress)
# test_ds = datasets.ImageFolder(os.path.join(BASE_DIR, 'test'), transform=transform)
# test_ds.samples = [s for s in tqdm(test_ds.samples, desc="🗂 Filtering dataset", ncols=100) if s[0] not in corrupted_files]
# test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# print("📂 Classes:", test_ds.classes)
# print("📑 Class-to-Index Mapping:", test_ds.class_to_idx)
# print(f"📸 Total test images (after filtering): {len(test_ds)}")

# # ───────────────────────────────────────
# # Load Model
# # ───────────────────────────────────────
# print("⏳ Loading checkpoint...")
# start = time.time()
# state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# print(f"📅 Loaded in {time.time() - start:.2f}s")

# model = create_model('xception', pretrained=False)
# in_features = model.fc.in_features
# model.fc = torch.nn.Sequential(
#     torch.nn.Dropout(0.4),
#     torch.nn.Linear(in_features, 1)
# )

# missing, unexpected = model.load_state_dict(state['model_state'], strict=False)
# if missing or unexpected:
#     print("⚠️ Mismatch in model state_dict:")
#     print("Missing keys:", missing)
#     print("Unexpected keys:", unexpected)

# model = model.to(DEVICE)
# model.eval()

# # ───────────────────────────────────────
# # Inference (with progress bar)
# # ───────────────────────────────────────
# print("🚀 Starting inference...")
# all_preds = []
# all_labels = []

# try:
#     with torch.no_grad():
#         for imgs, labels in tqdm(test_loader, desc="🔍 Testing", ncols=100, leave=True):
#             imgs = imgs.to(DEVICE)
#             outputs = model(imgs)
#             preds = (torch.sigmoid(outputs).squeeze() > 0.5).int().cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels.numpy())
# except Exception as e:
#     print("💥 Inference failed:", e)
#     exit(1)

# # ───────────────────────────────────────
# # Results
# # ───────────────────────────────────────
# print("\n✅ Inference done!")
# print("🎯 Accuracy: {:.2f}%".format(accuracy_score(all_labels, all_preds) * 100))
# print("\n🧾 Classification Report:\n")
# print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

# cm = confusion_matrix(all_labels, all_preds)
# print("📉 Confusion Matrix:\n")
# print(cm)

# # ───────────────────────────────────────
# # Save Confusion Matrix Plot
# # ───────────────────────────────────────
# def plot_and_save_confusion_matrix(cm, classes, save_path="confusion_matrix.png", normalize=False):
#     plt.figure(figsize=(6, 5))
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         fmt = ".2f"
#         title = "Xceptionnet Normalized Confusion Matrix"
#     else:
#         fmt = "d"
#         title = "Xceptionnet Confusion Matrix"

#     sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
#                 xticklabels=classes, yticklabels=classes,
#                 cbar=False, square=True, linewidths=0.5)

#     plt.title(title)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
#     os.makedirs("logs", exist_ok=True)
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     final_path = os.path.join("logs", f"conf_matrix_{timestamp}.png")
#     plt.savefig(final_path)
#     plt.close()
#     print(f"🖼️ Confusion matrix image saved to: {final_path}")

# plot_and_save_confusion_matrix(cm, test_ds.classes, normalize=False)

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
from timm import create_model
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import sys

# ─────────────────────────────
# 🔧 Config
# ─────────────────────────────
torch.cuda.set_device(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

BASE_DIR = r'E:\project\datasets\ds_5'
TEST_DIR = os.path.join(BASE_DIR, 'test')
CHECKPOINT_DIR = r'E:/project/models/checkpoints_xception_v1.0'
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'xception_epoch_11_v1.0.pth')
IMAGE_SIZE = 512
BATCH_SIZE = 20

# ─────────────────────────────
# 🎨 CLAHE Transform
# ─────────────────────────────
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

# ─────────────────────────────
# 📦 Load Dataset with Error Handling
# ─────────────────────────────
if not os.path.exists(TEST_DIR):
    print(f"❌ Test directory not found: {TEST_DIR}")
    sys.exit(1)

transform = transforms.Compose([
    CLAHE(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

try:
    test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print(f"❌ Failed to load test dataset: {e}")
    sys.exit(1)

# Dataset info
total_images = len(test_ds)
class_counts = {cls: 0 for cls in test_ds.classes}
for _, label in test_ds.samples:
    class_counts[test_ds.classes[label]] += 1

print(f"📊 Total test images: {total_images}")
for cls, count in class_counts.items():
    print(f"   {cls}: {count}")

# ─────────────────────────────
# 🧠 Load Model with Error Handling
# ─────────────────────────────
if not os.path.exists(BEST_CKPT_PATH):
    print(f"❌ Checkpoint file not found: {BEST_CKPT_PATH}")
    sys.exit(1)

try:
    model = create_model('xception', pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1)
    )
    model = model.to(DEVICE)

    checkpoint = torch.load(BEST_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✅ Loaded best checkpoint (Epoch {checkpoint['epoch']}, Val Acc = {checkpoint['val_acc']:.4f})")
except Exception as e:
    print(f"❌ Failed to load model checkpoint: {e}")
    sys.exit(1)

# ─────────────────────────────
# 🔍 Testing
# ─────────────────────────────
y_true = []
y_pred = []

try:
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
except Exception as e:
    print(f"❌ Error during testing: {e}")
    sys.exit(1)

# ─────────────────────────────
# 📉 Confusion Matrix & Classification Report
# ─────────────────────────────
try:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=test_ds.classes, digits=4)
    print("\n📄 Classification Report:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_ds.classes)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("XceptionNet Confusion Matrix - Test Set")
    save_path = os.path.join(CHECKPOINT_DIR, 'confusion_matrix_xception_test.png')
    plt.savefig(save_path)
    plt.close()

    print(f"📁 Confusion matrix saved at: {save_path}")
except Exception as e:
    print(f"❌ Failed to generate confusion matrix or report: {e}")
    sys.exit(1)

print("✅ Testing complete.")
