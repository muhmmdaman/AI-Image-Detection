import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "real_vs_ai_resnet18_final.pth"
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes=1):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes)
    )
    return model

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at '{MODEL_PATH}' — train it first.")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = build_model().to(device)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

class_names = checkpoint.get("class_names", ["real", "ai_generated"])
model.eval()
print(f"✅ Model loaded on {device}")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])

if len(sys.argv) < 2:
    print("Usage: python pred_resnet_learn.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    sys.exit(1)

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
    logits = model(img_tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().item()

pred_idx = 1 if prob >= 0.5 else 0
confidence = prob if pred_idx == 1 else (1 - prob)

print("\n--- 🧠 Prediction Result ---")
print(f"📁 File: {image_path}")
print(f"🔹 Prediction: {class_names[pred_idx].upper()} (class {pred_idx})")
print(f"🔹 Confidence: {confidence * 100:.2f}% (raw={prob:.4f})")
print("(0 → 'real', 1 → 'ai_generated')")

# --- USER FEEDBACK ---
if sys.stdin and sys.stdin.isatty():
    feedback = input("\nWas this prediction correct? (y/n): ").strip().lower()

    if feedback == "y":
        print("👍 Great! Model prediction confirmed.")
        sys.exit(0)

    elif feedback == "n":
        correct_label = input("Enter correct label ('r' for real, 'a' for ai'): ").strip().lower()
        if correct_label not in ["r", "a"]:
            print("⚠️ Invalid label. Skipping learning.")
            sys.exit(0)

        correct_value = 0.0 if correct_label == "r" else 1.0

        print("🧠 Updating model using your feedback...")
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        labels_tensor = torch.tensor([[correct_value]], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(img_tensor)
            loss = criterion(output, labels_tensor)
        loss.backward()
        optimizer.step()

        model.eval()

        new_state = {
            "model_state": model.state_dict(),
            "class_names": class_names
        }
        torch.save(new_state, MODEL_PATH)
        print(f"✅ Model updated and saved to '{MODEL_PATH}' (loss={loss.item():.6f})")

    else:
        print("⚠️ Invalid input. No learning applied.")
