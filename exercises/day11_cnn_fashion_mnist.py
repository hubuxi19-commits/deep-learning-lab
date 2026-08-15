import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)

# 1. 下载并加载 Fashion-MNIST
transform = transforms.ToTensor()

data_dir = Path(__file__).resolve().parent.parent / "data"

train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# 2. 定义 CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. 打印各主要层输出 shape
images, labels = next(iter(train_loader))
images = images.to(device)

with torch.no_grad():
    x = images
    print("\n输入图片 shape：", x.shape)

    x = model.features[0](x)
    print("第 1 个卷积层后：", x.shape)

    x = model.features[2](x)
    print("第 1 个池化层后：", x.shape)

    x = model.features[3](x)
    print("第 2 个卷积层后：", x.shape)

    x = model.features[5](x)
    print("第 2 个池化层后：", x.shape)

    x = torch.flatten(x, start_dim=1)
    print("展平后：", x.shape)

    logits = model(images)
    print("最终 logits：", logits.shape)


# 4. 训练与评估
epochs = 5
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(train_dataset)
    train_losses.append(average_loss)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

    print(
        f"第 {epoch + 1} 轮，"
        f"train loss={average_loss:.4f}，"
        f"test accuracy={test_accuracy:.4f}"
    )


# 5. 保存训练曲线
results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("CNN training loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, marker="o", color="tomato")
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.title("CNN test accuracy")
plt.grid(True)

plt.tight_layout()

output_path = results_dir / "day11_cnn_training.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\n训练图已保存到：{output_path}")