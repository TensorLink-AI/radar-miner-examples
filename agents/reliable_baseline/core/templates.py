"""Dynamic fallback code generator — builds a minimal valid model from the challenge.

When the LLM is unavailable or fails all retries, this generates a simple but
structurally correct model that passes validation and stays within FLOPs budget.
All parameters are derived from the challenge dict — nothing is hardcoded.
"""

from core.history import extract_flops_budget


def generate_fallback_code(challenge: dict) -> str:
    """Generate minimal valid model code from the challenge dict.

    Reads task_params to derive the build_model signature and model structure.
    Uses self-sizing to fit within the FLOPs budget.
    Returns empty string if challenge lacks enough info to generate valid code.
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    if not tp:
        return ""

    param_names = list(tp.keys())
    param_str = ", ".join(param_names)

    _, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 100_000

    # Detect ts_forecasting shape pattern:
    #   Input:  (batch, context_len, num_variates)
    #   Output: (batch, prediction_len, num_variates, len(quantiles))
    if all(k in tp for k in ("context_len", "prediction_len", "num_variates", "quantiles")):
        return _ts_forecasting_fallback(param_str, tp, target_flops)

    # Generic fallback: simple MLP that passes structural validation
    return _generic_fallback(param_str, tp, target_flops)


def _ts_forecasting_fallback(param_str: str, tp: dict, target_flops: int) -> str:
    """Generate a simple linear model for time-series forecasting tasks."""
    context_len = int(tp["context_len"])
    prediction_len = int(tp["prediction_len"])
    num_variates = max(int(tp["num_variates"]), 1)
    quantiles = tp["quantiles"]
    n_quantiles = len(quantiles) if isinstance(quantiles, list) else int(quantiles)

    # FLOPs for this architecture per forward pass:
    #   encoder: 2 * context_len * hidden_dim * num_variates
    #   decoder: 2 * hidden_dim * (prediction_len * n_quantiles) * num_variates
    # Solve for hidden_dim from target_flops:
    denom = 2 * num_variates * (context_len + prediction_len * n_quantiles)
    hidden_dim = max(4, target_flops // max(denom, 1))

    return f'''import torch
import torch.nn as nn


class FallbackForecaster(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.num_variates = num_variates
        self.n_quantiles = n_quantiles
        hidden_dim = {hidden_dim}
        self.encoder = nn.Linear(context_len, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        batch = x.shape[0]
        x = x.transpose(1, 2)  # (batch, num_variates, context_len)
        x = self.encoder(x)    # (batch, num_variates, hidden_dim)
        x = self.relu(x)
        x = self.decoder(x)    # (batch, num_variates, prediction_len * n_quantiles)
        x = x.view(batch, self.num_variates, self.prediction_len, self.n_quantiles)
        x = x.transpose(1, 2)  # (batch, prediction_len, num_variates, n_quantiles)
        return x


def build_model({param_str}):
    n_q = len(quantiles) if isinstance(quantiles, list) else int(quantiles)
    return FallbackForecaster(context_len, prediction_len, num_variates, n_q)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
'''


def _generic_fallback(param_str: str, tp: dict, target_flops: int) -> str:
    """Generate a generic minimal model for unknown task types."""
    # Use first integer param as input dim, second as output dim
    int_params = [(k, v) for k, v in tp.items() if isinstance(v, int)]
    if len(int_params) < 2:
        return ""

    input_key, input_dim = int_params[0]
    output_key, output_dim = int_params[1]

    # FLOPs: 2 * input_dim * hidden + 2 * hidden * output_dim
    denom = 2 * (input_dim + output_dim)
    hidden_dim = max(4, target_flops // max(denom, 1))

    return f'''import torch
import torch.nn as nn


class FallbackModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_model({param_str}):
    return FallbackModel({input_key}, {output_key}, {hidden_dim})


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
'''
