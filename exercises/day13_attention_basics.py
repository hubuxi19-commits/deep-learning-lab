import math
import torch
import torch.nn.functional as F


# 为了每次运行结果一致
torch.manual_seed(42)

# 假设一句话中有 3 个 token。
# 每一行代表一个 token 的初始向量（embedding）。
# shape: [token数量, 向量维度] = [3, 2]
X = torch.tensor([
    [1.0, 0.0],  # token 0
    [0.0, 1.0],  # token 1
    [1.0, 1.0],  # token 2
])

# 在真实 Transformer 中，这三个矩阵会通过训练自动学到。
# 今天先手动写出来，便于观察。
W_q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

W_k = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

W_v = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

# 1. 从同一个输入 X 生成 Q、K、V
# shape 都是 [3, 2]
Q = X @ W_q
K = X @ W_k
V = X @ W_v

print("X：")
print(X)

print("\nQ：")
print(Q)

print("\nK：")
print(K)

print("\nV：")
print(V)

# 2. 计算每个 Query 与每个 Key 的相似度
# K.T：把 K 从 [3, 2] 转为 [2, 3]
# Q @ K.T 的 shape 是 [3, 3]
# 行：当前正在处理哪个 token
# 列：当前 token 关注哪个 token
d = Q.shape[-1]
scores = Q @ K.T / math.sqrt(d)

print("\n注意力分数 scores：")
print(scores)
print("scores 的形状：", scores.shape)

# 3. 把分数转换为权重
# dim=-1 表示对每一行分别做 softmax
# 所以每一行的权重和都是 1
attention_weights = F.softmax(scores, dim=-1)

print("\n注意力权重 attention_weights：")
print(attention_weights)

print("\n每一行权重之和：")
print(attention_weights.sum(dim=-1))

# 4. 使用权重，对所有 V 做加权求和
# shape: [3, 3] @ [3, 2] -> [3, 2]
output = attention_weights @ V

print("\n最终注意力输出 output：")
print(output)
print("output 的形状：", output.shape)