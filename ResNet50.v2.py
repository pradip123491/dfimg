# ResNet50 using timm with CHUNK-wise loading
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
from PIL import Image

# ──────────────────────────────────────────────────
# 🔧 Configuration
# ──────────────────────────────────────────────────
torch.cuda.set_device(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

BASE_DIR = '/home/shavak/YC/project/datasets/ds_5'
IMAGE_SIZE = 512
BATCH_SIZE = 16
CHUNK_SIZE = 15000
EPOCHS = 20
LEARNING_RATE = 5e-5
CHECKPOINT_DIR = '/home/shavak/YC/project/models/checkpoints_resnet50_v2.0'
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
LOG_FILE_PATH = os.path.join(CHECKPOINT_DIR, 'training_log.txt')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
# Transforms
# ──────────────────────────────────────────────────
basic_transform = transforms.Compose([
    CLAHE(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────────
# 🧠 Chunked Dataset
# ──────────────────────────────────────────────────
class ChunkedImageFolder(Dataset):
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

train_ds = ChunkedImageFolder(os.path.join(BASE_DIR, 'train'), transform=basic_transform)
val_ds = ChunkedImageFolder(os.path.join(BASE_DIR, 'val'), transform=basic_transform)

print("📂 Classes:", train_ds.classes)
print("📁 Class-to-Index Mapping:", train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ──────────────────────────────────────────────────
# 🧠 Load ResNet50
# ──────────────────────────────────────────────────
model = create_model('resnet50', pretrained=True, num_classes=1)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.get_classifier().in_features, 1)
)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ───────────────────────────────────────────
# Resume from checkpoint
# ───────────────────────────────────────────
start_epoch = 0
best_val_acc = 0
if os.path.exists(LAST_CKPT_PATH):
    print("⏪ Resuming from last checkpoint...")
    checkpoint = torch.load(LAST_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint.get('val_acc', 0)
    print(f"🔁 Resumed from epoch {start_epoch}, val_acc = {best_val_acc:.4f}")

# ───────────────────────────────────────────
# 🔁 Training Loop
# ───────────────────────────────────────────
patience = 5
trigger_times = 0

with open(LOG_FILE_PATH, 'a') as log_file:
    for epoch in range(start_epoch, EPOCHS):
        try:
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

            for imgs, labels in pbar:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = torch.tensor(labels).float().unsqueeze(1).to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
                correct += (preds == labels.int().squeeze()).sum().item()
                total += labels.size(0)
                running_loss += loss.item()

                acc = 100 * correct / total
                pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{acc:.2f}%")

            # Validation
            model.eval()
            val_correct, val_total, val_loss_total = 0, 0, 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    labels = torch.tensor(labels).float().unsqueeze(1).to(DEVICE, non_blocking=True)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
                    val_correct += (preds == labels.int().squeeze()).sum().item()
                    val_total += labels.size(0)
                    val_loss_total += loss.item()

            val_acc = val_correct / val_total
            val_loss = val_loss_total / val_total
            scheduler.step(val_acc)

            log_str = f"📊 Epoch {epoch+1} → Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}\n"
            print(log_str)
            log_file.write(log_str)
            log_file.flush()

            # Save every epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, os.path.join(CHECKPOINT_DIR, f'resnet50_epoch_{epoch + 1}_v2.0.pth'))

            # Save for resume
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, LAST_CKPT_PATH)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                trigger_times = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_acc': val_acc
                }, os.path.join(CHECKPOINT_DIR, 'best_resnet50_v2.0.pth'))
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("⏹️ Early stopping triggered.")
                    log_file.write("⏹️ Early stopping triggered.\n")
                    break

        except Exception as e:
            error_str = f"[Epoch {epoch+1}] 🚨 Exception: {str(e)}\n"
            print(error_str)
            log_file.write(error_str)
            log_file.flush()
            continue

print("✅ Training complete.")
