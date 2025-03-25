# Vision Transformers (ViTs) with PyTorch — Cheat Sheet

## 1. ✅ Import Convention

```python
python
CopyEdit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

```

---

## 2. ⚙️ Core Functions / Classes / Concepts Table

| Name | Example Usage | Description |
| --- | --- | --- |
| `nn.Linear` | `nn.Linear(768, 768)` | Fully connected layer used in attention and MLP blocks. |
| `nn.MultiheadAttention` | `nn.MultiheadAttention(embed_dim=768, num_heads=8)` | Computes self-attention in the transformer encoder. |
| `nn.LayerNorm` | `nn.LayerNorm(768)` | Normalizes across the features; crucial for transformer stability. |
| `nn.TransformerEncoderLayer` | `nn.TransformerEncoderLayer(d_model=768, nhead=8)` | One encoder block: MSA + MLP + LayerNorm + Residuals. |
| `nn.TransformerEncoder` | `nn.TransformerEncoder(layer, num_layers=12)` | Stacks multiple encoder layers. |
| `nn.Parameter` | `self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))` | Learnable parameter — used for [CLS] tokens and position embeddings. |
| `torch.flatten()` | `x = x.flatten(2).transpose(1, 2)` | Flattens image patches before embedding. |
| `torch.nn.functional.softmax()` | `F.softmax(logits, dim=-1)` | Turns logits into probabilities. |
| `nn.Conv2d()` | Used in hybrid ViTs | Used to extract features from images (e.g., patchify using Conv). |

---

## 3. 🔁 Common Operations & Their Usage

### 📦 3.1 Patch Embedding

```python
python
CopyEdit
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

```

### 🧱 3.2 Add Class Token & Position Embeddings

```python
python
CopyEdit
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

def forward_features(x):
    x = self.patch_embed(x)  # shape: (B, N, D)
    B, N, _ = x.shape
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed

```

### ⚡ 3.3 Transformer Encoder Block

```python
python
CopyEdit
encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
x = transformer(x)

```

### 🔮 3.4 Classification Head

```python
python
CopyEdit
self.mlp_head = nn.Sequential(
    nn.LayerNorm(embed_dim),
    nn.Linear(embed_dim, num_classes)
)

def forward(self, x):
    x = self.forward_features(x)
    cls_token = x[:, 0]
    return self.mlp_head(cls_token)

```

---

## 4. 🚀 Useful Tips / Best Practices

- **Use `nn.Parameter`** for learnable `[CLS]` token and position embeddings.
- **Normalize patches** using `transforms.Normalize` before feeding into ViT.
- Use **pretrained ViTs (e.g., from Hugging Face or torchvision.models)** if you're not training from scratch.
- **Batch Size & LR**: Use `batch_size=64+`, and try `lr=3e-4` with warmup for stability.
- Use **gradient clipping** with `torch.nn.utils.clip_grad_norm_()` if gradients explode.
- If using your own patchification, make sure `img_size % patch_size == 0`.
- Set **`torch.backends.cudnn.benchmark = True`** for speedup on fixed-size inputs.

---

## 5. 🔗 Integration (Optional)

### With NumPy

```python
python
CopyEdit
img_np = np.random.rand(224, 224, 3)
img_tensor = torch.tensor(img_np).permute(2, 0, 1).float().unsqueeze(0)

```

### With DataLoader

```python
python
CopyEdit
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
loader = DataLoader(CustomDataset(transform=transform), batch_size=32, shuffle=True)

```

---

## 6. 🛠️ Mini Project / Small Exercises

### 🔥 Practice Tasks

1. **Patchify an image manually** using slicing and reshape operations.
2. **Implement `PatchEmbed`** module using `Conv2d` and `flatten`.
3. **Add learnable `[CLS]` token and position embeddings** to the patch embeddings.
4. **Build a single encoder block** using `nn.MultiheadAttention` and `LayerNorm`.
5. **Stack multiple encoder blocks** using `nn.Sequential` or `nn.TransformerEncoder`.
6. **Visualize position embeddings** using PCA or t-SNE.
7. **Fine-tune a pretrained ViT** on CIFAR-10 or Oxford Pets using `torchvision.models.vit_b_16`.
8. **Compare performance** of ResNet18 and ViT on the same dataset.
9. **Augment your dataset** using `RandomCrop`, `ColorJitter`, and test ViT robustness.
10. **Implement class attention visualization** using attention weights from `[CLS]` token.