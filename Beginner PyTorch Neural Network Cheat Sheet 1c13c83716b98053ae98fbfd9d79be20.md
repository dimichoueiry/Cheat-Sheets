# Beginner PyTorch Neural Network Cheat Sheet

## 1. ✅ Import Convention

```python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

```

---

## 2. 🧩 Core Functions/Classes/Concepts Table

| Function/Concept | Example Usage | Description |
| --- | --- | --- |
| `torch.tensor()` | `x = torch.tensor([1, 2, 3], dtype=torch.float32)` | Creates a tensor |
| `nn.Module` | `class Net(nn.Module): ...` | Base class for all models |
| `forward()` | `def forward(self, x):` | Defines how the data flows through the model |
| `nn.Linear(in, out)` | `nn.Linear(784, 128)` | Fully connected layer |
| `nn.ReLU()` | `nn.ReLU()` | Applies ReLU activation |
| `nn.CrossEntropyLoss()` | `loss_fn = nn.CrossEntropyLoss()` | Common classification loss |
| `optim.SGD` | `optimizer = optim.SGD(model.parameters(), lr=0.01)` | Optimizer for weight updates |
| `.backward()` | `loss.backward()` | Computes gradients |
| `.step()` | `optimizer.step()` | Updates weights based on gradients |
| `.zero_grad()` | `optimizer.zero_grad()` | Clears old gradients |

---

## 3. 🛠️ Common Operations & Their Usage

### 🔄 Data Preprocessing

```python

# Example dataset
X = torch.rand(100, 10)
y = torch.randint(0, 2, (100,))

# Dataset and DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

```

### 🧱 Model Building

```python

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()

```

### 🧮 Loss & Optimizer

```python

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

### 🔁 Training Loop

```python

for epoch in range(10):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```

### ✅ Evaluation

```python

with torch.no_grad():
    y_pred = model(X)
    predicted = torch.argmax(y_pred, dim=1)
    accuracy = (predicted == y).float().mean()

```

---

## 4. 🚀 Tips, Pro Tips, & Best Practices

| Tip | Why It Matters |
| --- | --- |
| Use `.float()` and `.long()` wisely | PyTorch layers expect `float32`, `CrossEntropyLoss` expects `int64` |
| Always call `model.train()` and `model.eval()` | Sets correct behavior for layers like Dropout/BatchNorm |
| Detach tensors before converting to NumPy | `tensor.detach().cpu().numpy()` prevents tracking gradients |
| Avoid gradient accumulation | Call `optimizer.zero_grad()` every training step |
| Use `with torch.no_grad()` during inference | Reduces memory usage and speeds up computation |
| Use `DataLoader` for batching and shuffling | Efficient input pipeline |

---

## 5. 🔌 Optional: Integration With NumPy / Pandas

```python

import pandas as pd
import numpy as np

# Convert from Pandas to Torch
df = pd.read_csv("data.csv")
X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

# From Torch to NumPy
x_np = X.numpy()

```

---

## 6. 💡 Mini Project: Binary Classification Pipeline

### 🧪 Practice Tasks

1. **Create a dataset** using `torch.rand()` and `torch.randint()` for binary classification.
2. **Build a 3-layer NN** with ReLU activations and softmax at the end.
3. **Write a full training loop** with batching, training, and evaluation logic.
4. **Track loss per epoch** and print it.
5. **Evaluate accuracy** after training on the full dataset.
6. **Convert predictions to NumPy** and plot confusion matrix.
7. **Add dropout** to prevent overfitting.
8. **Use SGD and Adam**, compare performance.
9. **Save and load the model** using `torch.save()` and `torch.load()`.
10. **Try GPU training**: use `model.to("cuda")`, if available.