"""Day 1：环境检查——确认 Python 和 PyTorch 可用。"""
import sys
import torch

print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print("环境就绪")