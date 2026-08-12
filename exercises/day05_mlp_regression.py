import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


torch.manual_seed(42)

n = 200
x = torch.linspace(-3, 3, n).reshape(-1, 1)
y = torch.sin(2 * x)

# Forward
model = nn.Sequential(
    nn.Linear(1,20),
    nn.ReLU(),
    nn.Linear(20,1)
)

# Loss
criterion = nn.MSELoss()

# Update
optimizer = optim.SGD(model.parameters(),lr = 0.05)


epochs = 1000
losses = []

for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if(epoch + 1) % 100 == 0:
        print(f"第 {epoch + 1} 轮， loss = {loss.item():.4f}")


with torch.no_grad():
    y_final = model(x)
    
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x.numpy(), y.numpy(), label="true: sin(2x)")
plt.plot(x.numpy(), y_final.numpy(), label = "MLP prediction")
plt.legend()
plt.title("True curve vs MLP prediction")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training loss")

plt.tight_layout()
plt.show()