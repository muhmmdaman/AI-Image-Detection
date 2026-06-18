import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4
PATIENCE = 7
MODEL_PATH = "real_vs_ai_resnet18.pth"
FINAL_MODEL_PATH = "real_vs_ai_resnet18_final.pth"
base_dir = "dataset"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🟢 Using device: {device}")

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomPerspective(distortion_scale=0.2)
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

print("📂 Loading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f"Detected classes: {class_names}")

counts = Counter(train_dataset.targets)
neg, pos = counts.get(0, 1), counts.get(1, 1)
pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
print(f"Class balance -> real: {neg}, ai: {pos}, pos_weight={pos_weight.item():.2f}")

def build_model(num_classes=1):
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes)
    )
    return model

model = build_model().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

start_epoch = 0
best_val_acc = 0.0
best_val_loss = float("inf")
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if os.path.exists(MODEL_PATH):
    print(f"🔄 Found existing checkpoint: '{MODEL_PATH}'. Resuming training...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✅ Resumed from epoch {start_epoch} (best val acc={best_val_acc:.3f})")
    except Exception as e:
        print(f"⚠️ Failed to resume checkpoint: {e}")
else:
    print("🆕 Starting new training run...")

print(f"🚀 Training for up to {EPOCHS} epochs (starting from {start_epoch})...\n")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total += y.size(0)
        correct += (preds == y).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        for x, y in val_loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total += y.size(0)
            correct += (preds == y).sum().item()
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    val_loss /= total
    val_acc = correct / total
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

    if (epoch + 1) % 5 == 0:
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_names": class_names,
            "pos_weight": pos_weight.item(),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "epoch": epoch
        }
        torch.save(best_state, MODEL_PATH)
        torch.save(best_state, FINAL_MODEL_PATH)
        print(f"💾 Model improved (Val Acc: {val_acc:.3f}) — saved at epoch {epoch+1}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"\n⛔ Early stopping triggered after {epoch+1} epochs (no improvement for {PATIENCE}).")
        break

if "best_state" in locals():
    torch.save(best_state, MODEL_PATH)
    torch.save(best_state, FINAL_MODEL_PATH)
    print(f"\n✅ Final best model saved as '{MODEL_PATH}' and '{FINAL_MODEL_PATH}'")
else:
    print("⚠️ No model was saved — no validation improvement detected.")

try:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend(), plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend(), plt.title("Loss")
    plt.tight_layout()
    plt.savefig("training_curves_resnet18.png")
    print("📈 Saved training curves as training_curves_resnet18.png")
except Exception as e:
    print(f"Plot error: {e}")
