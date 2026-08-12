import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path


# 1. 固定随机种子：每次运行得到相同结果，方便复现
torch.manual_seed(42)


# 2. 生成两类二维数据
n_per_class = 200

class_0 = torch.randn(n_per_class, 2) * 0.7 + torch.tensor([0.0, 0.0])
class_1 = torch.randn(n_per_class, 2) * 0.7 + torch.tensor([2.5, 2.5])

X = torch.cat([class_0, class_1], dim=0).float()
y = torch.cat([
    torch.zeros(n_per_class, dtype=torch.long),
    torch.ones(n_per_class, dtype=torch.long),
], dim=0)

print("全部数据 X 的形状：", X.shape)
print("全部标签 y 的形状：", y.shape)


# 3. 打乱并划分训练集 / 测试集：80% / 20%
indices = torch.randperm(len(X))
split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("训练集 X_train 的形状：", X_train.shape)
print("测试集 X_test 的形状：", X_test.shape)


# 4. 定义分类模型
# 输入：两个特征
# 输出：两个 logits，分别对应类别 0 和类别 1
model = nn.Linear(2, 2)

# CrossEntropyLoss 直接接收 logits，不要手动加 softmax
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)


# 5. 训练模型
epochs = 300
losses = []

for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"第 {epoch + 1:3d} 轮，loss = {loss.item():.4f}")


# 6. 在测试集评估
with torch.no_grad():
    test_logits = model(X_test)
    test_probabilities = torch.softmax(test_logits, dim=1)
    y_pred = torch.argmax(test_logits, dim=1)

    accuracy = (y_pred == y_test).float().mean().item()

    # 二分类混淆矩阵：
    # 行是真实类别，列是预测类别
    tn = ((y_test == 0) & (y_pred == 0)).sum().item()
    fp = ((y_test == 0) & (y_pred == 1)).sum().item()
    fn = ((y_test == 1) & (y_pred == 0)).sum().item()
    tp = ((y_test == 1) & (y_pred == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

print("\n===== 测试集结果 =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 score : {f1:.4f}")

print("\n混淆矩阵（行=真实标签，列=预测标签）")
print(torch.tensor([
    [tn, fp],
    [fn, tp],
]))

print("\n前 5 个样本的 logits：")
print(test_logits[:5])

print("\n前 5 个样本的类别概率：")
print(test_probabilities[:5])

print("\n前 5 个样本：真实标签 / 预测标签")
print(y_test[:5])
print(y_pred[:5])


# 7. 保存两张图：原始数据分布 + 训练 loss
results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(
    X_test[y_test == 0, 0],
    X_test[y_test == 0, 1],
    color="royalblue",
    label="true class 0",
)
plt.scatter(
    X_test[y_test == 1, 0],
    X_test[y_test == 1, 1],
    color="tomato",
    label="true class 1",
)
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Test data")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses, color="purple")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training loss")
plt.grid(True)

plt.tight_layout()

output_path = results_dir / "day06_classification_training.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\n训练图已保存到：{output_path}")