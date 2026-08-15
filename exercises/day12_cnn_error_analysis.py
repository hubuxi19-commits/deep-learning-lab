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

data_dir = Path(__file__).resolve().parent.parent / "data"
results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root=data_dir, train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=data_dir, train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. 训练 CNN
epochs = 5

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

        total_loss += loss.item() * labels.size(0)

    print(f"第 {epoch + 1} 轮，train loss={total_loss / len(train_dataset):.4f}")


# 2. 收集所有测试集预测
model.eval()

all_images = []
all_true_labels = []
all_pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(device))
        predictions = torch.argmax(logits, dim=1).cpu()

        all_images.append(images)
        all_true_labels.append(labels)
        all_pred_labels.append(predictions)

all_images = torch.cat(all_images)
all_true_labels = torch.cat(all_true_labels)
all_pred_labels = torch.cat(all_pred_labels)

accuracy = (all_true_labels == all_pred_labels).float().mean().item()
print(f"\n测试集准确率：{accuracy:.4f}")


# 3. 构造混淆矩阵：行=真实类别，列=预测类别
confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

for true_label, pred_label in zip(all_true_labels, all_pred_labels):
    confusion_matrix[true_label, pred_label] += 1

print("\n混淆矩阵（行=真实类别，列=预测类别）")
print(confusion_matrix)

print("\n类别索引：")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# 4. 找出错分图片
wrong_indices = torch.where(all_true_labels != all_pred_labels)[0]

print(f"\n错分图片数量：{len(wrong_indices)}")

# 保存前 12 张错分图片
plt.figure(figsize=(12, 9))

for plot_index, image_index in enumerate(wrong_indices[:12]):
    image = all_images[image_index, 0]
    true_label = all_true_labels[image_index].item()
    pred_label = all_pred_labels[image_index].item()

    plt.subplot(3, 4, plot_index + 1)
    plt.imshow(image, cmap="gray")
    plt.title(
        f"true: {class_names[true_label]}\npred: {class_names[pred_label]}",
        fontsize=9,
    )
    plt.axis("off")

plt.suptitle("First 12 misclassified Fashion-MNIST images", fontsize=14)
plt.tight_layout()

error_path = results_dir / "day12_misclassified_examples.png"
plt.savefig(error_path, dpi=150)
plt.show()

print(f"\n错分样本图已保存到：{error_path}")