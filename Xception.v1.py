import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from timm import create_model

# ─────────────────────────────
# 🔧 Configuration
# ─────────────────────────────
torch.cuda.set_device(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

BASE_DIR = '/home/shavak/YC/project/datasets/ds_5'
IMAGE_SIZE = 512
BATCH_SIZE = 20
EPOCHS = 20
LEARNING_RATE = 5e-5
CHECKPOINT_DIR = '/home/shavak/YC/project/models/checkpoints_xception_v1.0'
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
# Transforms
# ─────────────────────────────
transform = transforms.Compose([
    CLAHE(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────
# 📁 Load Dataset
# ─────────────────────────────
train_ds = datasets.ImageFolder(os.path.join(BASE_DIR, 'train'), transform=transform)
val_ds = datasets.ImageFolder(os.path.join(BASE_DIR, 'val'), transform=transform)

print("📂 Classes:", train_ds.classes)
print("📑 Class-to-Index Mapping:", train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────
# 🧠 Load Xception Model (from timm)
# ─────────────────────────────
model = create_model('xception', pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 1)
)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ─────────────────────────────
# Resume if checkpoint exists
# ─────────────────────────────
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

# ─────────────────────────────
# 🔁 Training Loop
# ─────────────────────────────
patience = 5
trigger_times = 0

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
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
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step(val_acc)
    print(f"📊 Epoch {epoch+1} → Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}")

    # Save every epoch
    torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_acc': val_acc
    }, os.path.join(CHECKPOINT_DIR, f'xception_epoch_{epoch + 1}_v1.0.pth'))

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
        }, os.path.join(CHECKPOINT_DIR, 'best_xception_v1.0.pth'))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("⏹️ Early stopping triggered.")
            break

print("✅ Training complete.")
