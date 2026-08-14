import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)

# 固定同一份圆形数据，所有实验共用
n = 1000
X = torch.rand(n, 2) * 4 - 2
distance = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
y = (distance > 1.0).long()

indices = torch.randperm(n) 
split = 800
train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]


def build_model():
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def run_experiment(name, optimizer_name, lr, epochs=200):
    # 每组从相同的初始权重开始，比较才公平
    torch.manual_seed(123)
    model = build_model()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for _ in range(epochs):
        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = torch.argmax(test_logits, dim=1)
        test_accuracy = (test_pred == y_test).float().mean().item()

    print(
        f"{name:12s} "
        f"final loss={losses[-1]:.4f}, "
        f"test accuracy={test_accuracy:.4f}"
    )

    return name, losses, test_accuracy


experiments = [
    run_experiment("SGD lr=0.001", "SGD", 0.001),
    run_experiment("SGD lr=0.01", "SGD", 0.01),
    run_experiment("SGD lr=0.1", "SGD", 0.1),
    run_experiment("Adam lr=0.001", "Adam", 0.001),
    run_experiment("Adam lr=0.01", "Adam", 0.01),
    run_experiment("Adam lr=0.1", "Adam", 0.1),
]

results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 5))

for name, losses, _ in experiments:
    plt.plot(losses, label=name)

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Optimizer and learning-rate comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = results_dir / "day10_optimizer_comparison.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\n图已保存到：{output_path}")