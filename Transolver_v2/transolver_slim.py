#!/usr/bin/env python3
"""
Transolver-Slim —— 面向 RKPM 权重回归的精简 Transolver

与 `physicsnemo.models.transolver.Transolver` 的关系
----------------------------------------------------
本文件是**自包含**的忠实重实现（standard Transolver，非 ++、非结构化网格分支），
逐层对齐原版计算图：

    preprocess(Mlp)  →  N×TransolverBlock  →  末层 block 输出 out_dim
    Block:  fx = Attn(LN(fx)) + fx ;  fx = MLP(LN(fx)) + fx ;  末层再接 LN+Linear
    PhysicsAttention(IrregularMesh):
        x_mid, fx_mid = Linear_x(x), Linear_fx(x)                → (B,N,H,D)
        slice_weights = softmax(Linear_slice(x_mid) / T, dim=-1) → (B,N,H,S)
        slice_token   = Σ_N (slice_weights / Σ_N slice_weights) · fx_mid → (B,H,S,D)
        out_slice     = SDPA(q,k,v = Linear_qkv(slice_token))    → (B,H,S,D)
        out           = Linear_out( Σ_S slice_weights · out_slice )

改这一版的两个原因
------------------
1. **去依赖**：原版需要 physicsnemo（并连带 warp / s3fs / jaxtyping / einops），
   干净环境里常装不上。本文件只依赖 torch。
2. **可控容量**：原版默认 `n_hidden=128, n_layers=4, slice_num=32, mlp_ratio=4`
   在本任务上是 **778,641 参数**，而训练集只有约 2,000~6,000 个支撑笼
   （笼内 27 点由同一个 3 维相位刚性决定，**不是**独立样本）。参数量是独立样本数的
   两个数量级，这是"过拟合"质疑的直接靶子。本文件把 `n_hidden / n_layers /
   slice_num / mlp_ratio` 全部暴露为旋钮，并给出三档预设。

预设
----
| 预设   | n_hidden | n_layers | n_head | slice_num | mlp_ratio |
| :----- | -------: | -------: | -----: | --------: | --------: |
| `orig` |      128 |        4 |      4 |        32 |         4 |
| `slim` |       48 |        2 |      4 |         8 |         2 |
| `tiny` |       32 |        2 |      4 |         4 |         2 |

精确参数量运行 `python3 transolver_slim.py` 自检打印。若 `slim` 与 `orig` 精度相当，
即可直接反驳"靠大容量记忆"，并顺带压低推理时延（原版约 5.8 ms/10k点）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TransolverSlim", "PRESETS", "build_transolver", "count_params"]


PRESETS = {
    "orig": dict(n_hidden=128, n_layers=4, n_head=4, slice_num=32, mlp_ratio=4),
    "slim": dict(n_hidden=48, n_layers=2, n_head=4, slice_num=8, mlp_ratio=2),
    "tiny": dict(n_hidden=32, n_layers=2, n_head=4, slice_num=4, mlp_ratio=2),
}

_ACT = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}


class Mlp(nn.Module):
    """两层前馈，对应原版 `_TransolverMlp`。"""

    def __init__(self, in_features, hidden_features, out_features, act="gelu", dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = _ACT[act]()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class PhysicsAttentionIrregularMesh(nn.Module):
    """非结构化点云上的物理注意力（对应原版 `PhysicsAttentionIrregularMesh`）。

    把 N 个 token 软分配到 S 个"物理切片"，在切片之间做注意力，再投影回 token 空间。
    复杂度 O(N·S + S²) 而非 O(N²)——本任务 N=27、S=4~32。
    """

    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0, slice_num=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim, self.heads, self.dim_head, self.slice_num = dim, heads, dim_head, slice_num

        # 可学习 softmax 温度：与原版一致，初始化 0.5、前向时 clamp 到 [0.5, 5]
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.qkv_project = nn.Linear(dim_head, dim_head * 3)
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):                                   # x: (B, N, C)
        B, N, _ = x.shape
        H, D, S = self.heads, self.dim_head, self.slice_num

        x_mid = self.in_project_x(x).view(B, N, H, D)
        fx_mid = self.in_project_fx(x).view(B, N, H, D)

        proj = self.in_project_slice(x_mid)                 # (B, N, H, S)
        temp = torch.clamp(self.temperature, min=0.5, max=5.0)
        slice_weights = torch.softmax(proj / temp, dim=-1)  # (B, N, H, S)

        # 先归一再聚合，避免低精度下溢出（与原版同一写法）
        slice_norm = slice_weights.sum(dim=1) + 1e-2        # (B, H, S)
        normed = slice_weights / slice_norm[:, None, :, :]  # (B, N, H, S)
        # (B,H,S,N) @ (B,H,N,D) -> (B,H,S,D)
        slice_token = torch.matmul(normed.permute(0, 2, 3, 1), fx_mid.permute(0, 2, 1, 3))

        qkv = self.qkv_project(slice_token).view(B, H, S, 3, D)
        q, k, v = qkv.unbind(3)                             # 各 (B, H, S, D)
        out_slice = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # (B,N,H,S) × (B,H,S,D) -> (B,N,H,D)
        out_x = torch.einsum("bnhs,bhsd->bnhd", slice_weights, out_slice)
        out_x = out_x.reshape(B, N, H * D)
        return self.out_dropout(self.out_linear(out_x))


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act, mlp_ratio,
                 out_dim, slice_num, last_layer=False):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"n_hidden({hidden_dim}) 必须能被 n_head({num_heads}) 整除")
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttentionIrregularMesh(
            dim=hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln_mlp1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Mlp(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, act, dropout),
        )
        if last_layer:
            self.ln_mlp2 = nn.Sequential(nn.LayerNorm(hidden_dim),
                                         nn.Linear(hidden_dim, out_dim))

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.ln_mlp1(fx) + fx
        return self.ln_mlp2(fx) if self.last_layer else fx


class TransolverSlim(nn.Module):
    """精简版 Transolver。

    forward 签名与原版保持一致：`model(fx=..., embedding=...)`，
    因此既有训练/推理代码可直接替换，无需改调用处。

    输入 `[B, N, functional_dim]` + `[B, N, embedding_dim]`，输出 `[B, N, out_dim]`。
    本任务两者都传 `rel_hat`（3 维），N=27，out_dim=1。
    """

    def __init__(self, functional_dim=3, out_dim=1, embedding_dim=3,
                 n_layers=2, n_hidden=48, dropout=0.0, n_head=4,
                 act="gelu", mlp_ratio=2, slice_num=8):
        super().__init__()
        self.cfg = dict(functional_dim=functional_dim, out_dim=out_dim,
                        embedding_dim=embedding_dim, n_layers=n_layers,
                        n_hidden=n_hidden, dropout=dropout, n_head=n_head,
                        act=act, mlp_ratio=mlp_ratio, slice_num=slice_num)

        self.preprocess = Mlp(functional_dim + embedding_dim, n_hidden * 2, n_hidden, act)
        self.blocks = nn.ModuleList([
            TransolverBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                            act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                            slice_num=slice_num, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, fx, embedding=None):
        if embedding is not None:
            fx = torch.cat((embedding, fx), dim=-1)   # 顺序与原版一致：embedding 在前
        fx = self.preprocess(fx)
        for blk in self.blocks:
            fx = blk(fx)
        return fx


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_transolver(preset="slim", functional_dim=3, out_dim=1, embedding_dim=3,
                     dropout=0.0, act="gelu", **overrides):
    """按预设名构建模型；`overrides` 可逐项覆盖预设（如 n_hidden=64）。"""
    if preset not in PRESETS:
        raise ValueError(f"未知预设 {preset}，可选 {list(PRESETS)}")
    cfg = dict(PRESETS[preset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return TransolverSlim(functional_dim=functional_dim, out_dim=out_dim,
                          embedding_dim=embedding_dim, dropout=dropout, act=act, **cfg)


if __name__ == "__main__":
    # 自检：三档预设的参数量与前向形状
    x = torch.randn(2, 27, 3)
    print(f"{'预设':6s} {'参数量':>10s}  配置")
    for name in PRESETS:
        m = build_transolver(name)
        y = m(fx=x, embedding=x)
        assert y.shape == (2, 27, 1), y.shape
        print(f"{name:6s} {count_params(m):10,d}  {PRESETS[name]}")
