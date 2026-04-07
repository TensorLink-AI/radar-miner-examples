"""Pre-validated model templates per size bucket — the safety net.

Each template is a complete Python module string that passes AST validation
and produces correct output shapes.  These are fallbacks for when the LLM
is unreachable or returns invalid code.

Architecture guidance:
  - tiny   (~275K FLOPs):  Simple linear mixer, no attention
  - small  (~1.1M FLOPs):  Lightweight conv + linear
  - medium_small (~5.5M):  Patch-based with small transformer
  - medium (~27M FLOPs):   Multi-layer transformer
  - large  (~69M FLOPs):   Full transformer with more heads/layers
"""

_TINY_TEMPLATE = '''\
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
            return x
        else:  # denorm
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class TinyLinearMixer(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.n_quantiles = n_quantiles
        self.revin = RevIN(num_variates)
        hidden = 24
        self.mix1 = nn.Linear(context_len, hidden)
        self.act = nn.GELU()
        self.mix2 = nn.Linear(hidden, prediction_len * n_quantiles)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        x = self.revin(x, "norm")
        h = x.transpose(1, 2)  # (batch, num_variates, context_len)
        h = self.mix1(h)       # (batch, num_variates, hidden)
        h = self.norm(h)
        h = self.act(h)
        h = self.mix2(h)       # (batch, num_variates, pred * n_q)
        b, v, _ = h.shape
        h = h.view(b, v, self.prediction_len, self.n_quantiles)
        h = h.permute(0, 2, 1, 3)  # (batch, pred, variates, n_q)
        h = self.revin(h.reshape(b, self.prediction_len, self.num_variates * self.n_quantiles), "denorm")
        return h.view(b, self.prediction_len, self.num_variates, self.n_quantiles)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return TinyLinearMixer(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)


def training_config():
    return {"batch_size": 64, "grad_accum_steps": 1, "grad_clip": 1.0, "eval_interval": 200}


def build_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
'''

_SMALL_TEMPLATE = '''\
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
            return x
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class SmallConvMixer(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.n_quantiles = n_quantiles
        self.revin = RevIN(num_variates)
        hidden = 36
        self.conv1 = nn.Conv1d(num_variates, hidden, kernel_size=7, padding=3)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden)
        self.linear1 = nn.Linear(context_len, prediction_len)
        self.norm2 = nn.LayerNorm(prediction_len)
        self.head = nn.Linear(hidden, num_variates * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        x = self.revin(x, "norm")
        h = x.permute(0, 2, 1)   # (batch, variates, context_len)
        h = self.conv1(h)         # (batch, hidden, context_len)
        h = h.permute(0, 2, 1)   # (batch, context_len, hidden)
        h = self.norm1(h)
        h = self.act(h)
        h = h.permute(0, 2, 1)   # (batch, hidden, context_len)
        h = self.linear1(h)       # (batch, hidden, prediction_len)
        h = h.permute(0, 2, 1)   # (batch, prediction_len, hidden)
        h = self.norm2(h)
        h = self.act(h)
        h = self.head(h)          # (batch, prediction_len, variates * n_q)
        b = h.shape[0]
        h = h.view(b, self.prediction_len, self.num_variates, self.n_quantiles)
        h_flat = h.reshape(b, self.prediction_len, self.num_variates * self.n_quantiles)
        h_flat = self.revin(h_flat, "denorm")
        return h_flat.view(b, self.prediction_len, self.num_variates, self.n_quantiles)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return SmallConvMixer(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def training_config():
    return {"batch_size": 64, "grad_accum_steps": 1, "grad_clip": 1.0, "eval_interval": 200}


def build_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
'''

_MEDIUM_SMALL_TEMPLATE = '''\
import torch
import torch.nn as nn
import math


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
            return x
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class PatchTransformer(nn.Module):
    """PatchTST-style model: patching + small transformer."""
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.n_quantiles = n_quantiles
        self.revin = RevIN(num_variates)

        patch_size = 16
        d_model = 64
        self.patch_size = patch_size
        n_patches = context_len // patch_size
        self.n_patches = n_patches

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model * n_patches, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        x = self.revin(x, "norm")
        b, L, V = x.shape
        # Channel-independent: process each variate separately
        x = x.permute(0, 2, 1)  # (batch, V, L)
        x = x.reshape(b * V, L)
        # Patch
        x = x.reshape(b * V, self.n_patches, self.patch_size)  # (b*V, n_patches, patch_size)
        x = self.patch_embed(x) + self.pos_embed  # (b*V, n_patches, d_model)
        x = self.encoder(x)  # (b*V, n_patches, d_model)
        x = x.reshape(b * V, -1)  # (b*V, n_patches * d_model)
        x = self.head(x)  # (b*V, prediction_len * n_q)
        x = x.reshape(b, V, self.prediction_len, self.n_quantiles)
        x = x.permute(0, 2, 1, 3)  # (batch, pred, V, n_q)
        x_flat = x.reshape(b, self.prediction_len, V * self.n_quantiles)
        x_flat = self.revin(x_flat, "denorm")
        return x_flat.view(b, self.prediction_len, V, self.n_quantiles)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return PatchTransformer(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def training_config():
    return {"batch_size": 32, "grad_accum_steps": 2, "grad_clip": 1.0, "eval_interval": 200}


def build_scheduler(optimizer, total_steps):
    warmup = total_steps // 10
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
'''

_MEDIUM_TEMPLATE = '''\
import torch
import torch.nn as nn
import math


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
            return x
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class MediumTransformer(nn.Module):
    """Multi-layer transformer with patching for medium bucket."""
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.n_quantiles = n_quantiles
        self.revin = RevIN(num_variates)

        patch_size = 16
        d_model = 128
        self.patch_size = patch_size
        n_patches = context_len // patch_size
        self.n_patches = n_patches

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * n_patches, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        x = self.revin(x, "norm")
        b, L, V = x.shape
        x = x.permute(0, 2, 1).reshape(b * V, L)
        x = x.reshape(b * V, self.n_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.reshape(b * V, -1)
        x = self.head(x)
        x = x.reshape(b, V, self.prediction_len, self.n_quantiles)
        x = x.permute(0, 2, 1, 3)
        x_flat = x.reshape(b, self.prediction_len, V * self.n_quantiles)
        x_flat = self.revin(x_flat, "denorm")
        return x_flat.view(b, self.prediction_len, V, self.n_quantiles)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return MediumTransformer(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)


def training_config():
    return {"batch_size": 32, "grad_accum_steps": 2, "grad_clip": 1.0, "eval_interval": 200}


def build_scheduler(optimizer, total_steps):
    warmup = total_steps // 10
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
'''

_LARGE_TEMPLATE = '''\
import torch
import torch.nn as nn
import math


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
            return x
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class LargeTransformer(nn.Module):
    """Full transformer for the large FLOPs bucket."""
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.n_quantiles = n_quantiles
        self.revin = RevIN(num_variates)

        patch_size = 16
        d_model = 192
        self.patch_size = patch_size
        n_patches = context_len // patch_size
        self.n_patches = n_patches

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=384,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.head = nn.Linear((d_model // 2) * n_patches, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        x = self.revin(x, "norm")
        b, L, V = x.shape
        x = x.permute(0, 2, 1).reshape(b * V, L)
        x = x.reshape(b * V, self.n_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = self.head_proj(x)
        x = self.act(x)
        x = x.reshape(b * V, -1)
        x = self.head(x)
        x = x.reshape(b, V, self.prediction_len, self.n_quantiles)
        x = x.permute(0, 2, 1, 3)
        x_flat = x.reshape(b, self.prediction_len, V * self.n_quantiles)
        x_flat = self.revin(x_flat, "denorm")
        return x_flat.view(b, self.prediction_len, V, self.n_quantiles)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return LargeTransformer(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


def training_config():
    return {"batch_size": 16, "grad_accum_steps": 4, "grad_clip": 1.0, "eval_interval": 200}


def build_scheduler(optimizer, total_steps):
    warmup = total_steps // 10
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
'''

_TEMPLATES = {
    "tiny": _TINY_TEMPLATE,
    "small": _SMALL_TEMPLATE,
    "medium_small": _MEDIUM_SMALL_TEMPLATE,
    "medium": _MEDIUM_TEMPLATE,
    "large": _LARGE_TEMPLATE,
}

# Default fallback when bucket is unknown
_DEFAULT_BUCKET = "small"


def get_template(bucket: str) -> str:
    """Return a complete, pre-validated Python module string for the given bucket."""
    return _TEMPLATES.get(bucket, _TEMPLATES[_DEFAULT_BUCKET])
