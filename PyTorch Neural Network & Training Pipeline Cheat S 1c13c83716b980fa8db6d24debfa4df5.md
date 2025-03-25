# PyTorch Neural Network & Training Pipeline Cheat Sheet (Intermediate–Advanced)

Here’s a **deep, practical PyTorch cheat sheet** tailored for your goal: building neural networks, running the full training pipeline (preprocessing → forward pass → loss → backprop → update), and integrating everything smartly.

---

## 1. **Import Convention**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

```

---

## 2. **Core Functions / Classes / Concepts Table**

| Name | Example Usage | Description |
| --- | --- | --- |
| `torch.tensor()` | `torch.tensor([1,2,3])` | Creates a tensor from a list or array. |
| `nn.Module` | `class MyModel(nn.Module):` | Base class for all neural network models. |
| `forward()` | `def forward(self, x):` | Defines forward computation of the model. |
| `nn.Linear` | `nn.Linear(10, 5)` | Fully connected layer: input 10 → output 5. |
| `nn.ReLU()` | `self.relu = nn.ReLU()` | Applies ReLU non-linearity. |
| `nn.CrossEntropyLoss()` | `loss_fn = nn.CrossEntropyLoss()` | Combines softmax + negative log-likelihood. |
| `optim.SGD()` | `optim.SGD(model.parameters(), lr=0.01)` | Stochastic Gradient Descent optimizer. |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` | Clears accumulated gradients. |
| `loss.backward()` | `loss.backward()` | Backpropagates gradients. |
| `optimizer.step()` | `optimizer.step()` | Updates weights using gradients. |
| `torch.no_grad()` | `with torch.no_grad():` | Disables gradient tracking (useful for inference). |

---

## 3. **Common Operations & Usage Snippets**

### 🔹 Data Prep & Loading

```python
# From NumPy
import numpy as np
x = torch.from_numpy(np.random.rand(100, 10)).float()
y = torch.from_numpy(np.random.randint(0, 2, 100)).long()

# Dataset + DataLoader
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

```

### 🔹 Define a Neural Network

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()

```

### 🔹 Training Loop

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```

### 🔹 Evaluation / Accuracy

```python
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total

```

---

## 4. **🔥 Tips / Best Practices**

| Tip | Why It Matters |
| --- | --- |
| Always use `.float()` and `.long()` correctly | `float` for features, `long` for class labels |
| Use `model.eval()` during inference | Disables dropout/batchnorm behavior |
| `torch.no_grad()` saves memory | Don’t track gradients when not training |
| Batch size = power of 2 (e.g., 32/64) | Optimized for GPU |
| Use `torch.save(model.state_dict(), "model.pth")` | Save only weights, not the whole class structure |
| Use weight initialization (Xavier, He) | Especially useful for custom models |
| Profile your code using `torch.utils.bottleneck` | Diagnose slow parts of training |

---

## 5. **(Optional) Integration with NumPy & Pandas**

```python
import pandas as pd
df = pd.read_csv('data.csv')
X = torch.tensor(df.drop('label', axis=1).values).float()
y = torch.tensor(df['label'].values).long()

dataset = TensorDataset(X, y)

```

- From NumPy to torch: `torch.from_numpy(np_array)`
- From torch to NumPy: `tensor.numpy()` *(only if `tensor.requires_grad == False`)*

---

## 6. **🔥 Mini Project: Quick Practice Tasks**

### 🧠 Level: Intermediate+ (Practice + Real World)

1. **Build a model for binary classification using tabular data (10 features).**
2. **Add dropout to the model to reduce overfitting.**
3. **Use `nn.BCELoss()` and sigmoid for binary outputs.**
4. **Write a function to plot training loss vs epochs.**
5. **Write a utility to evaluate top-1 accuracy on test set.**
6. **Switch from SGD to Adam and tune learning rate.**
7. **Visualize predictions using `matplotlib` (e.g., confusion matrix).**
8. **Train with GPU if available (use `.to(device)`)**
9. **Save and reload your model weights.**
10. **Wrap everything in a `train(model, loader)` and `evaluate(model, loader)` function for modularity.**

---