"""Day 3: Tensor 操作与自动求导。"""

import torch

# ========== 任务 1: 用 torch.tensor 复现 Day 2 矩阵操作 ==========
print("=== 任务 1: 用 torch.tensor 复现 Day 2 矩阵操作 ===")
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
b = torch.tensor([10., 20., 30.])
c = torch.tensor([[100.],
                  [200.],
                  [300.]])

print(f"A + b =\n{A + b}")
print(f"\nA @ c =\n{A @ c}")
print(f"\nsum(axis=0) = {A.sum(axis=0)}")
print(f"sum(axis=1) = {A.sum(axis=1)}")


# ========== 任务 2: 验证 autograd ==========
print("\n=== 任务 2: 验证 autograd ===")

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x

y.backward()

print(x.grad)
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = 2x + 3 = {2*2 + 3} (手算)")
print(f"x.grad = {x.grad.item()} (autograd)")
print(f"一致: {x.grad.item() == 7.0}")


# ========== 任务 3: 多变量求导 ==========
print("\n=== 任务 3: 多变量求导 ===")
w = torch.tensor([1.0, 2.0, 3.0], requires_grad= True)
loss = (w**2).sum()
loss.backward()

print(f"w = {w.tolist()}")
print(f"loss = {loss.item()}")
print(f"w.grad = {w.grad.tolist()}")   # 应为 [2, 4, 6]
print(f"手算: [2*1, 2*2, 2*3] = [2, 4, 6]")


# ========== 任务 4（思考题，不写代码）==========
# 如果 loss 是一个 3 元素向量的 sum()，那么每个 wi 的梯度是什么？
