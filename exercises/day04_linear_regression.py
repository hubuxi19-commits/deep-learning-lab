"""Day 4: 从零实现线性回归。"""
import torch
import matplotlib.pyplot as plt

# ===== 1. 生成数据 =====
torch.manual_seed(42)           # 固定随机种子结果可复现
n = 100                         # 100个数据点
x = torch.randn(n,1)            # shape （100， 1），随机输入
true_w = 3.0
true_b = 2.0
y_true = true_w * x + true_b + 0.5 * torch.randn(n, 1)  # 真值 + 高斯噪声

'''
print(f"x shape: {x.shape}")
print(f"y_true shape: {y_true.shape}")
print(f"前3个 x: {x[:3].flatten().tolist()}")
print(f"前3个 y: {y_true[:3].flatten().tolist()}")
'''

# 版本 B: 预测和损失
# ===== 2. 初始化参数 =====
w = torch.tensor(0.0, requires_grad = True)
b = torch.tensor(0.0, requires_grad = True)
lr = 0.1            # 学习率
epochs = 100        # 训练轮数

losses = []         # 记录每轮 loss, 用来画图
w_history = []      # 记录 w 的变化
b_history = []      # 记录 b 的变化

'''
y_pred = w * x + b
loss = ((y_pred - y_true) ** 2).mean()
loss.backward()

print(f"\n初始化 w=0, b=0 时：")
print(f"loss = {loss.item():.2f}")
print(f"w.grad = {w.grad.item():.2f}")
print(f"b.grad = {b.grad.item():.2f}")
'''

# ===== 3. 训练循环 =====
for epoch in range(epochs):
    # Forward: 预测
    y_pred = w * x + b

    # Loss: 均方误差
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward: 求梯度
    loss.backward()

    # Update: 调参数（用 torch.no_grad() 包裹，因为更新不算在计算图里）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 梯度清零！ 注意这是最容易被漏掉的一步
    w.grad.zero_()
    b.grad.zero_()

    # 记录
    losses.append(loss.item())
    w_history.append(w.item())
    b_history.append(b.item())


# ===== 4. 打印最终结果 =====
print(f"真实参数: w={true_w}, b={true_b}")
print(f"学到的参数: w={w.item():.3f}, b={b.item():.3f}")
print(f"最终 loss: {losses[-1]:.4f}")


# ===== 5. 画图 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 左：loss 曲线
axes[0].plot(losses)
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# 中：w 的变化
axes[1].plot(w_history)
axes[1].axhline(y=true_w, color='r', linestyle='--', label=f'true w={true_w}')
axes[1].set_title("w vs Epoch")
axes[1].legend()

# 右：b 的变化
axes[2].plot(b_history)
axes[2].axhline(y=true_b, color='r', linestyle='--', label=f'true b={true_b}')
axes[2].set_title("b vs Epoch")
axes[2].legend()

plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
results_dir = os.path.join(project_dir, "results")
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "day04_training.png"))
plt.show()

print("\n图片已保存到 results/day04_training.png")