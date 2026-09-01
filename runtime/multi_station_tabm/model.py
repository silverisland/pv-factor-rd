from __future__ import annotations

import random
from typing import Any

import numpy as np
import rtdl_num_embeddings
import tabm
import torch

from .config import Config


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def configure_reproducibility(seed: int, config: Config) -> dict[str, Any]:
    deterministic = dict(config["training"].get("deterministic", {}))
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 3)
    cudnn_deterministic = bool(deterministic.get("cudnn_deterministic", True))
    cudnn_benchmark = bool(deterministic.get("cudnn_benchmark", False))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.use_deterministic_algorithms(
        bool(deterministic.get("use_deterministic_algorithms", False))
    )
    return {
        "seed": seed,
        "python_seed": seed,
        "numpy_seed": seed + 1,
        "torch_seed": seed + 2,
        "cuda_seed": seed + 3,
        "cuda_manual_seed_all": bool(torch.cuda.is_available()),
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "deterministic_algorithms": bool(
            deterministic.get("use_deterministic_algorithms", False)
        ),
    }


def model_contract(n_features: int) -> dict[str, Any]:
    return {
        "library": "tabm",
        "constructor": "tabm.TabM.make",
        "n_num_features": n_features,
        "cat_cardinalities": [],
        "d_out": 1,
        "num_embeddings": "rtdl_num_embeddings.LinearReLUEmbeddings",
        "architecture_kwargs": {},
        "ensemble_reduction": "mean_over_k",
    }


def make_model(n_features: int, device: torch.device) -> torch.nn.Module:
    """Match the supplied tabm4pv.py model initialization exactly."""
    embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_features)
    return tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        num_embeddings=embeddings,
    ).to(device)


@torch.inference_mode()
def predict_normalized(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(values, device=device)
    indices = torch.arange(len(tensor), device=device)
    outputs = []
    for batch in indices.split(batch_size):
        outputs.append(model(tensor[batch], None).squeeze(-1).float())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    return torch.cat(outputs).cpu().numpy().mean(axis=1).astype(np.float32)
