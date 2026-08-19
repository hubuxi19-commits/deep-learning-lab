import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, token数, embed_dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        d = Q.shape[-1]

        # [batch, token数, embed_dim]
        # @ [batch, embed_dim, token数]
        # -> [batch, token数, token数]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d)

        attention_weights = F.softmax(scores, dim=-1)

        # [batch, token数, token数]
        # @ [batch, token数, embed_dim]
        # -> [batch, token数, embed_dim]
        attention_output = attention_weights @ V

        return attention_output, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        self.attention = SingleHeadSelfAttention(embed_dim)

        # 注意力后的残差连接与归一化
        self.norm1 = nn.LayerNorm(embed_dim)

        # FFN：对每个 token 独立执行的小型 MLP
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # FFN 后的残差连接与归一化
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 1. token 之间交流
        attention_output, attention_weights = self.attention(x)

        # 2. 残差连接 + LayerNorm
        x_after_attention = self.norm1(x + attention_output)

        # 3. 每个 token 独立进行非线性变换
        ffn_output = self.ffn(x_after_attention)

        # 4. 第二次残差连接 + LayerNorm
        output = self.norm2(x_after_attention + ffn_output)

        return output, attention_weights


torch.manual_seed(42)

# 2 个样本；每个样本 4 个 token；每个 token 是 8 维向量
x = torch.randn(2, 4, 8)

model = TransformerBlock(
    embed_dim=8,
    hidden_dim=32,
)

output, attention_weights = model(x)

print("输入 x 的形状：", x.shape)
print("注意力权重的形状：", attention_weights.shape)
print("Transformer Block 输出的形状：", output.shape)

print("\n第 1 个样本的注意力权重：")
print(attention_weights[0])

print("\n第 1 个样本每个 token 输出向量的均值：")
print(output[0].mean(dim=-1))

print("\n第 1 个样本每个 token 输出向量的标准差：")
print(output[0].std(dim=-1, unbiased=False))