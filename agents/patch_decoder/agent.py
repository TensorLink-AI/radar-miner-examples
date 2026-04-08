"""Patch Decoder — deterministic agent that emits a RevIN patch-MLP decoder
scaled to the centre of the requested FLOPs bucket.

No LLM calls.  The architecture is fully parameterised by a per-bucket
scaling table, so every run with the same challenge is perfectly reproducible.

Architecture:
  1. RevIN normalisation (learnable affine)
  2. Non-overlapping patch embedding (Linear: patch_size -> d_model)
  3. Learnable positional embedding
  4. N stacked MLP blocks (Linear -> LayerNorm -> GELU -> Linear residual)
  5. Flatten patches -> projection head -> (pred_len, num_variates, n_quantiles)
  6. RevIN denormalisation
"""

import sys
import tempfile
import textwrap

from core import validation, history


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ── Scaling table ────────────────────────────────────────────────
# Each entry targets the *centre* of its FLOPs bucket.
# Keys: d_model, n_layers, ff_mult, patch_size, lr, batch_size, grad_accum
_SCALING = {
    "tiny": {
        "d_model": 32,
        "n_layers": 1,
        "ff_mult": 2,
        "patch_size": 16,
        "lr": "5e-4",
        "batch_size": 64,
        "grad_accum": 1,
    },
    "small": {
        "d_model": 48,
        "n_layers": 1,
        "ff_mult": 2,
        "patch_size": 16,
        "lr": "3e-4",
        "batch_size": 64,
        "grad_accum": 1,
    },
    "medium_small": {
        "d_model": 64,
        "n_layers": 2,
        "ff_mult": 2,
        "patch_size": 16,
        "lr": "3e-4",
        "batch_size": 32,
        "grad_accum": 2,
    },
    "medium": {
        "d_model": 128,
        "n_layers": 3,
        "ff_mult": 2,
        "patch_size": 16,
        "lr": "2e-4",
        "batch_size": 32,
        "grad_accum": 2,
    },
    "large": {
        "d_model": 192,
        "n_layers": 4,
        "ff_mult": 2,
        "patch_size": 16,
        "lr": "1e-4",
        "batch_size": 16,
        "grad_accum": 4,
    },
}


def _generate_code(cfg: dict) -> str:
    """Render a complete Python module string from scaling config."""
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]
    ff = d * cfg["ff_mult"]
    ps = cfg["patch_size"]
    lr = cfg["lr"]
    bs = cfg["batch_size"]
    ga = cfg["grad_accum"]

    return textwrap.dedent(f"""\
        import torch
        import torch.nn as nn
        import math


        class RevIN(nn.Module):
            \"\"\"Reversible Instance Normalization.\"\"\"
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


        class MLPBlock(nn.Module):
            \"\"\"Pre-norm MLP block with residual connection.\"\"\"
            def __init__(self, d_model, ff_dim, dropout=0.1):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.fc1 = nn.Linear(d_model, ff_dim)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(ff_dim, d_model)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                residual = x
                x = self.norm(x)
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.drop(x)
                return x + residual


        class PatchDecoder(nn.Module):
            \"\"\"Patch-based MLP decoder with RevIN.\"\"\"
            def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
                super().__init__()
                self.prediction_len = prediction_len
                self.num_variates = num_variates
                self.n_quantiles = n_quantiles

                self.revin = RevIN(num_variates)

                patch_size = {ps}
                d_model = {d}
                self.patch_size = patch_size
                n_patches = context_len // patch_size
                self.n_patches = n_patches

                # Patch embedding
                self.patch_embed = nn.Linear(patch_size, d_model)
                self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

                # MLP blocks
                self.blocks = nn.Sequential(
                    *[MLPBlock(d_model, {ff}, dropout=0.1) for _ in range({n_layers})]
                )
                self.out_norm = nn.LayerNorm(d_model)

                # Projection head
                self.head = nn.Linear(d_model * n_patches, prediction_len * n_quantiles)

            def forward(self, x):
                # x: (batch, context_len, num_variates)
                x = self.revin(x, "norm")
                b, L, V = x.shape

                # Channel-independent: process each variate separately
                x = x.permute(0, 2, 1).reshape(b * V, L)
                x = x.reshape(b * V, self.n_patches, self.patch_size)
                x = self.patch_embed(x) + self.pos_embed
                x = self.blocks(x)
                x = self.out_norm(x)
                x = x.reshape(b * V, -1)
                x = self.head(x)

                x = x.reshape(b, V, self.prediction_len, self.n_quantiles)
                x = x.permute(0, 2, 1, 3)  # (batch, pred, V, n_q)
                x_flat = x.reshape(b, self.prediction_len, V * self.n_quantiles)
                x_flat = self.revin(x_flat, "denorm")
                return x_flat.view(b, self.prediction_len, V, self.n_quantiles)


        def build_model(context_len, prediction_len, num_variates, quantiles):
            return PatchDecoder(context_len, prediction_len, num_variates, len(quantiles))


        def build_optimizer(model):
            return torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay=0.01)


        def training_config():
            return {{"batch_size": {bs}, "grad_accum_steps": {ga}, "grad_clip": 1.0, "eval_interval": 200}}


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
    """)


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness.  Deterministic — no LLM calls."""

    # ── Identify bucket ──────────────────────────────────────────
    flops_min, flops_max = history.extract_flops_budget(challenge)
    bucket = history.identify_bucket(flops_min, flops_max)
    center_flops = (flops_min + flops_max) // 2

    _log(f"[patch_decoder] Bucket: {bucket}, FLOPs range: {flops_min:,}-{flops_max:,}, "
         f"center: {center_flops:,}")

    # ── Select scaling config ────────────────────────────────────
    cfg = _SCALING.get(bucket, _SCALING["medium"])

    # ── Generate code ────────────────────────────────────────────
    code = _generate_code(cfg)

    ok, errors = validation.validate_code(code, challenge)
    if not ok:
        _log(f"[patch_decoder] BUG: generated code failed validation: {errors}")

    name = f"patch_decoder_{bucket}"
    motivation = (
        f"Deterministic RevIN patch-MLP decoder for {bucket} bucket "
        f"(d_model={cfg['d_model']}, layers={cfg['n_layers']}, "
        f"target ~{center_flops:,} FLOPs)"
    )

    # ── Update scratchpad ────────────────────────────────────────
    scratch_dir = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected global
    except Exception:
        pass

    state = history.load_state(scratch_dir) if scratch_dir else {}
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=center_flops, strategy="deterministic",
    )
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected global
    except Exception:
        pass

    return {"code": code, "name": name, "motivation": motivation}
