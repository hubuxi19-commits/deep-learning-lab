"""Day 5: 用 torch.nn 重写线性回归。"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ===== 1. 数据（和 Day 4 完全一样） =====
torch.manual_seed(42)
n = 100
x = torch.randn(n,1)
true_w = 3.0
true_b = 2.0
y_true = true_w * x + true_b + 0.5 * torch.randn(n, 1)

# ===== 2. 三行定义模型、损失、优化器 =====
model = nn.Linear(1,1)       # 输入 1 维，输出 1 维 -> y = wx + b
criterion = nn.MSELoss()     # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr = 0.1)   # 随机梯度下降


# ===== 3. 训练循环 =====
epochs = 100
losses = []

for epoch in range(epochs):
    # Forward
    y_pred = model(x)

    # Loss
    loss = criterion(y_pred, y_true)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    losses.append(loss.item())

# ===== 4. 结果 =====
print(f"真实参数: w={true_w}, b={true_b}")
print(f"学到的参数: w={model.weight.item():.3f}, b={model.bias.item():.3f}")
print(f"最终 loss: {losses[-1]:.4f}")

# ===== 5. 画图 =====
plt.plot(losses)
plt.title("Day 5: Loss vs Epoch (torch.nn)")
plt.xlabel("Epoch")
plt.ylabel("Loss")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
results_dir = os.path.join(project_dir, "results")
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "day05_training.png"))
plt.show()