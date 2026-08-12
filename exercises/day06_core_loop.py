import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.tensor([
    [1.0, 1.0],
    [1.5, 1.2],
    [0.8, 1.6],
    [4.0, 4.0],
    [4.5, 3.8],
    [3.7, 4.4],
])

y = torch.tensor([0, 0, 0, 1, 1, 1])

model = nn.Linear(2, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("最终 loss：", loss.item())

with torch.no_grad():
    logits = model(X)
    y_pred = torch.argmax(logits, dim=1)

print("真实标签：", y)
print("预测标签：", y_pred)
