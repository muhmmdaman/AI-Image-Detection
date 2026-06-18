import os
import io
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from torchvision import models, transforms
from PIL import Image
import shutil

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
app.secret_key = "super_secret_key_123"
USER_FILE = "users.json"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../real_vs_ai_resnet18_final.pth"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    try:
        with open(USER_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
        return {}

def save_users(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_model():
    print("🔁 Loading model...")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(DEVICE)
    model.eval()

    class_names = checkpoint.get("class_names", ["real", "ai_generated"])
    print(f"✅ Model loaded with classes: {class_names}")
    return model, class_names

model, CLASS_NAMES = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return send_from_directory(app.static_folder, "login.html")

    email = request.form.get("email").strip().lower()
    password = request.form.get("password")
    users = load_users()

    if email in users and users[email]["password"] == password:
        session["user"] = email
        return redirect(url_for("index"))
    return "<h3>❌ Invalid credentials. <a href='/login'>Try again</a></h3>"

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return send_from_directory(app.static_folder, "register.html")

    email = request.form.get("email").strip().lower()
    password = request.form.get("password")
    users = load_users()

    if email in users:
        return "<h3>⚠️ User already exists. <a href='/login'>Login</a></h3>"

    users[email] = {"password": password}
    save_users(users)
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


def generate_heatmap(model, input_tensor, original_image, save_path):
    model.eval()
    target_layer = model.layer4[-1]
    gradients, activations = [], []

    def forward_hook(module, input, output):
        activations.append(output.detach())
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[:, 0].backward(retain_graph=True)

    grad = gradients[0].cpu().numpy()[0]
    act = activations[0].cpu().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * act, axis=0), 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (original_image.width, original_image.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    fwd.remove()
    bwd.remove()
    return save_path


@app.route("/predict", methods=["POST"])
def predict():
    global model, CLASS_NAMES
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
        prob = torch.sigmoid(output).squeeze().item()

        if not CLASS_NAMES or len(CLASS_NAMES) < 2:
            CLASS_NAMES = ["real", "ai_generated"]

        label_idx = 1 if prob >= 0.5 else 0
        label = CLASS_NAMES[label_idx]
        confidence = prob if label_idx == 1 else (1 - prob)

        heatmap_path = os.path.abspath(os.path.join(app.static_folder, "assets", "heatmap.png"))
        generate_heatmap(model, input_tensor, img, heatmap_path)

        print(f"✅ Prediction: {label} ({confidence*100:.2f}%)")
        return jsonify({
            "label": label,
            "confidence": round(confidence, 4),
            "heatmap": "/assets/heatmap.png"
        })

    except Exception as e:
        print("Prediction error:", e)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    global model, CLASS_NAMES
    try:
        if "file" not in request.files or "label" not in request.form:
            return jsonify({"error": "File and label required"}), 400

        label_str = request.form["label"].strip().lower()
        file = request.files["file"]
        y = 0.0 if "real" in label_str else 1.0

        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        for param in model.parameters(): param.requires_grad = False
        for param in model.fc.parameters(): param.requires_grad = True

        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-5, weight_decay=1e-6)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        optimizer.zero_grad()
        labels = torch.tensor([[y]], dtype=torch.float32, device=DEVICE)
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        model.eval()

        torch.save({
            "model_state": model.state_dict(),
            "class_names": CLASS_NAMES
        }, MODEL_PATH)

        model, CLASS_NAMES = load_model()
        model.to(DEVICE)
        model.eval()

        return jsonify({"message": f"Model retrained successfully ({label_str})"})
    except Exception as e:
        print("Feedback error:", e)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    print("🚀 Running AI Image Detector on http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, debug=True)
