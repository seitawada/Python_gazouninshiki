import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# ===== 設定 =====
DATA_DIR = "images4"  # ← データセットのパスここを変更
EPOCHS = 20     # ← 学習回数ここを変更
BATCH_SIZE = 16
LR = 0.0003

# ===== 前処理（精度アップ版） =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ===== データ =====
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes
print("クラス:", class_names)

# ===== モデル（転移学習） =====
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 学習 =====
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

# ===== 保存 =====
torch.save({
    "model": model.state_dict(),
    "classes": class_names
}, "model5.pth")

print("✅ model5.pth 保存完了")
