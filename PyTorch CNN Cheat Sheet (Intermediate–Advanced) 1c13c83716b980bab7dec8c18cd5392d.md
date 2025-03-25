# PyTorch CNN Cheat Sheet (Intermediate–Advanced)

## 🧩 1. Import Convention

```python
python
CopyEdit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

```

---

## 🔧 2. Core Functions/Classes/Concepts Table

| **Name** | **Example Usage** | **Description** |
| --- | --- | --- |
| `nn.Conv2d` | `nn.Conv2d(1, 32, kernel_size=3)` | 2D convolution layer (in_channels, out_channels, kernel_size) |
| `nn.MaxPool2d` | `nn.MaxPool2d(2, 2)` | Max pooling over 2D input |
| `nn.ReLU` | `nn.ReLU(inplace=True)` | ReLU activation function |
| `nn.Flatten` | `nn.Flatten()` | Flattens input for fully connected layer |
| `nn.Linear` | `nn.Linear(512, 10)` | Fully connected layer |
| `nn.Sequential` | `nn.Sequential(...)` | Stack layers in order without custom class |
| `F.relu` | `F.relu(x)` | Functional ReLU; used in `forward()` |
| `model(x)` | `output = model(img)` | Performs forward pass |
| `model.eval()` | `model.eval()` | Switches model to evaluation mode (no dropout/batchnorm updates) |
| `model.train()` | `model.train()` | Switches back to training mode |
| `optimizer.step()` | `optimizer.step()` | Updates weights |
| `loss.backward()` | `loss.backward()` | Computes gradients via backprop |
| `torch.no_grad()` | `with torch.no_grad(): ...` | Disables gradient tracking during inference |
| `torchvision.transforms` | `transforms.ToTensor()` | Preprocessing images for CNNs |

---

## ⚙️ 3. Common Operations & Their Usage

### ✅ Data Preprocessing

```python
python
CopyEdit
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

```

### ✅ Define CNN Architecture

```python
python
CopyEdit
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)              # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 14x14 -> 14x14
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

### ✅ Training Loop

```python
python
CopyEdit
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

```

### ✅ Evaluation

```python
python
CopyEdit
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

```

---

## 💡 4. Useful Tips / Pro Tips / Best Practices

- **Use `nn.Sequential`** for simpler models to keep code clean.
- **Batch normalization** (`nn.BatchNorm2d`) helps stabilize training.
- **Don't forget `.train()` and `.eval()`** when switching between modes.
- **Flatten properly** before `nn.Linear`, use `.view()` or `nn.Flatten()`.
- **Avoid hardcoding image sizes** — use `x.shape` to make it dynamic.
- **Use GPU if available**:
    
    ```python
    python
    CopyEdit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    images, labels = images.to(device), labels.to(device)
    
    ```
    
- **Monitor overfitting** — use validation set and track loss/accuracy.
- **Visualize filters/activations** for debugging.

---

## 🔌 5. Optional Integration with NumPy, Pandas, OpenCV

```python
python
CopyEdit
# Convert PyTorch tensor to NumPy
np_img = image_tensor.numpy()

# Convert NumPy array to Tensor
tensor_img = torch.tensor(np_img)

# From Pandas to Tensor
torch_tensor = torch.tensor(df.values, dtype=torch.float32)

# From OpenCV image to Tensor
import cv2
img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
img_tensor = torch.tensor(img / 255.0).unsqueeze(0).unsqueeze(0).float()

```

---

## 🧪 6. Mini Project / Small Practical Exercises

> Try these to solidify your understanding of CNNs in PyTorch:
> 
1. ✅ Build a CNN to classify **FashionMNIST** dataset with similar architecture.
2. ✅ Add **BatchNorm2d** and observe training speed/stability.
3. ✅ Replace **ReLU** with **LeakyReLU** or **ELU**, compare performances.
4. ✅ Add a dropout layer (`nn.Dropout`) and observe its effect on overfitting.
5. ✅ Visualize filters after training: `model.conv1.weight.data`
6. ✅ Freeze the first convolutional layer (transfer learning style).
7. ✅ Write your own DataLoader for a custom folder of PNG images.
8. ✅ Convert your PyTorch model to TorchScript and save it.
9. ✅ Integrate your trained model into a Flask API or Streamlit app.
10. ✅ Use **Grad-CAM** or **torchcam** to visualize where your CNN is focusing.