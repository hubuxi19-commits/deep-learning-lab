import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)

# 1. 生成二维圆形数据
n = 1000
X = torch.rand(n, 2) * 4 - 2  # 每个坐标都在 [-2, 2] 内

# 距离原点较近为类别 0；较远为类别 1
distance = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
y = (distance > 1.0).long()

# 2. 划分训练集和测试集
indices = torch.randperm(n)
split = int(n * 0.8)

train_idx = indices[:split]
test_idx = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 3. MLP：Linear → ReLU → Linear
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

# 4. 训练
epochs = 500
losses = []

for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"第 {epoch + 1} 轮，loss = {loss.item():.4f}")

# 5. 测试集评估
with torch.no_grad():
    test_logits = model(X_test)
    y_pred = torch.argmax(test_logits, dim=1)
    accuracy = (y_pred == y_test).float().mean().item()

print(f"\n测试集准确率：{accuracy:.4f}")

# 6. 画真实类别、预测类别与损失曲线
results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            s=12, color="royalblue", label="true class 0")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            s=12, color="tomato", label="true class 1")
plt.title("True labels")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1],
            s=12, color="royalblue", label="predicted class 0")
plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1],
            s=12, color="tomato", label="predicted class 1")
plt.title("MLP predictions")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(losses, color="purple")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training loss")
plt.grid(True)

plt.tight_layout()

output_path = results_dir / "day08_mlp_circle_training.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"图片已保存到：{output_path}")