import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # 三个可学习的线性层：分别生成 Q、K、V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x 的形状：[batch_size, token数量, embedding维度]
        """

        # 1. 由输入生成 Q、K、V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. K 的最后两个维度互换
        # [batch, token数, dim] -> [batch, dim, token数]
        K_transpose = K.transpose(-2, -1)

        # 3. 算注意力分数
        d = Q.shape[-1]
        scores = Q @ K_transpose / math.sqrt(d)

        # 4. 每个 token 对全部 token 分配注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 5. 根据权重汇总 V
        output = attention_weights @ V

        return output, attention_weights


torch.manual_seed(42)

# 两句话；每句话 4 个 token；每个 token 用 8 维向量表示
x = torch.randn(2, 4, 8)

attention = SelfAttention(embed_dim=8)

output, attention_weights = attention(x)

print("输入 x 的形状：", x.shape)
print("注意力输出 output 的形状：", output.shape)
print("注意力权重 attention_weights 的形状：", attention_weights.shape)

print("\n第 1 句话的注意力权重：")
print(attention_weights[0])

print("\n第 1 句话每一行权重之和：")
print(attention_weights[0].sum(dim=-1))

print("\n模型中可学习参数的形状：")
for name, parameter in attention.named_parameters():
    print(name, parameter.shape)