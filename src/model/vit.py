import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=64, patch_size=8, in_channels=3, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(in_channels * patch_size * patch_size, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(B, -1, C * p * p)
        return self.projection(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class ViTBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=8, in_channels=3,
                 d_model=128, num_heads=4, num_layers=6, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
