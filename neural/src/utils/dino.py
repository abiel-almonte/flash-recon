import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 6
        self.head_dim = 64  # 384 // 6
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(384, 384 * 3, bias=True)
        self.proj = nn.Linear(384, 384, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, N, D
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, 384)
        return self.proj(x)


class LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)
        self.norm = nn.Identity()
        self.num_patches = (518 // 14) * (518 // 14)  # 37*37 = 1369

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B, 384, H/14, W/14
        x = x.flatten(2).transpose(1, 2)  # B, N, 384
        x = self.norm(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self) -> None:
        """
        Hardcoded configuration for vit small

        Parameters:
            embed_dim = 384
            patch_size = 14
            depth = 12
            num_heads = 6
            mlp_ratio = 4
            init_values = 1.0
        """
        super().__init__()
        self.embed_dim = 384
        self.patch_size = 14
        self.num_register_tokens = 0

        self.patch_embed = PatchEmbed()
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 384))

        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Identity()

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=False,
        )
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int],
        return_class_token: bool = False,
    ) -> tuple:
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in n:
                output.append(x)
        assert len(output) == len(n)

        outputs = [self.norm(out) for out in output]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])
