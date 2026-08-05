"""Day 2:NumPy 数组操作与线性代数基础练习。"""
import numpy as np

# === 任务 1：创建数组并打印 shape ===
# 创建一个 2 行 3 列的矩阵 A，值随意
# 创建一个长度为 3 的向量 b
# 创建一个 3 行 1 列的列向量 c
# 打印三者的 shape

A = np.array([[1,2,3],
              [4,5,6]])   # shape (2,3)
b = np.array([10,20,30])  # shape (3,)
c = np.array([[100],
              [200],
              [300]])     # shape (3,1)

print("=== 任务 1: 形状 ===")
print(f"A shape: {A.shape}")
print(f"b shape: {b.shape}")
print(f"c shape: {c.shape}")


# === 任务 2：广播 ===
# 把 (2, 3) 的 A 和 (3,) 的 b 相加
# 打印结果和结果的 shape
# 思考：b 是怎么被广播成能加 A 的？

print("\n=== 任务 2: 广播 ===")
result2 = A + b
print(f"A + b =\n {result2}")
print(f"Result shape: {result2.shape}")


# === 任务 3：矩阵乘法 ===
# 计算 A @ c，即 (2, 3) @ (3, 1)
# 打印结果和结果的 shape
# 用纸笔手算一遍，验证结果

print("\n=== 任务 3: 矩阵乘法 ===")
result3 = A @ c
print(f"A @ c =\n{result3}")
print(f"shape: {result3.shape}")


# === 任务 4：规约 ===
# 对 A 执行 sum(axis=0) 和 sum(axis=1)
# 打印结果和结果的 shape
# 解释 axis=0 和 axis=1 各自"消灭"了哪个维度

print("\n=== 任务 4: 规约 ===")
sum0 = A.sum(axis=0)
sum1 = A.sum(axis=1)
print(f"A.sum(axis=0) = {sum0}, shape: {sum0.shape}")
print(f"A.sum(axis=1) = {sum1}, shape: {sum1.shape}")


 # === 任务 5: 验证形状规则 ===

print("\n=== Task 5: Shape Mismatch ===")
try:
      D = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
      E = np.array([[1], [2]])                   # (2, 1)
      print(f"(3, 2) @ (2, 1) = \n{D @ E}")      # ✅ 2 == 2
      print(f"shape: {(D @ E).shape}")

      F = np.array([[1, 2, 3]])                  # (1, 3)
      D @ F.T                                     # 试图 (3, 2) @ (3, 1) ❌
except Exception as e:
      print(f"Error: {e}")