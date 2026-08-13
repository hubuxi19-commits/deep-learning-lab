import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)

# 1. 生成少量、带标签噪声的圆形分类数据
n = 160
X = torch.rand(n, 2) * 4 - 2
distance = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
y = (distance > 1.0).long()

# 故意翻转 15% 标签，模拟真实数据的标注噪声
noise_indices = torch.randperm(n)[:24]
y[noise_indices] = 1 - y[noise_indices]

# 2. 训练 / 验证切分
indices = torch.randperm(n)
split = 100

train_idx = indices[:split]
val_idx = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]


def build_model(dropout=0.0):
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 2),
    )


def train_and_evaluate(name, weight_decay=0.0, dropout=0.0):
    model = build_model(dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay,
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(500):
        model.train()

        train_logits = model(X_train)
        train_loss = criterion(train_logits, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)

            train_pred = torch.argmax(train_logits, dim=1)
            val_pred = torch.argmax(val_logits, dim=1)

            train_accuracy = (train_pred == y_train).float().mean().item()
            val_accuracy = (val_pred == y_val).float().mean().item()

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    print(
        f"{name:18s} "
        f"train acc={train_accuracies[-1]:.3f}, "
        f"val acc={val_accuracies[-1]:.3f}, "
        f"train loss={train_losses[-1]:.3f}, "
        f"val loss={val_losses[-1]:.3f}"
    )

    return {
        "name": name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }


# 3. 运行三组实验
experiments = [
    train_and_evaluate("no regularization"),
    train_and_evaluate("weight decay", weight_decay=0.01),
    train_and_evaluate("dropout", dropout=0.4),
]

# 4. 画损失和准确率曲线
results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for result in experiments:
    plt.plot(result["train_losses"], label=f'{result["name"]} train')
    plt.plot(result["val_losses"], "--", label=f'{result["name"]} val')
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training and validation loss")
plt.legend(fontsize=8)
plt.grid(True)

plt.subplot(1, 2, 2)
for result in experiments:
    plt.plot(result["train_accuracies"], label=f'{result["name"]} train')
    plt.plot(result["val_accuracies"], "--", label=f'{result["name"]} val')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and validation accuracy")
plt.legend(fontsize=8)
plt.grid(True)

plt.tight_layout()

output_path = results_dir / "day09_overfitting_comparison.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\n图已保存到：{output_path}")
