# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Cosmos3 manual CUDA-graph configuration gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import validate_manual_cuda_graphs
from vllm_omni.diffusion.offloader.config import OffloadStrategy


def _config(**overrides):
    base = {
        "parallel_config": SimpleNamespace(world_size=1, use_hsdp=False),
        "cache_backend": None,
        "quantization_config": None,
        "lora_path": None,
        "enable_sleep_mode": False,
        "enforce_eager": False,
        "diffusion_compile_granularity": "regional",
        "diffusion_attention_schedule": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _offload(strategy: OffloadStrategy = OffloadStrategy.NONE):
    return SimpleNamespace(strategy=strategy)


def test_single_gpu_regional_compile_is_accepted() -> None:
    validate_manual_cuda_graphs(_config(), torch.device("cuda"), _offload())
    validate_manual_cuda_graphs(_config(enforce_eager=True), torch.device("cuda"), _offload())
    validate_manual_cuda_graphs(_config(cache_backend="sea_cache"), torch.device("cuda"), _offload())


@pytest.mark.parametrize(
    "overrides",
    [
        {"parallel_config": SimpleNamespace(world_size=2, use_hsdp=False)},
        {"parallel_config": SimpleNamespace(world_size=1, use_hsdp=True)},
        {"cache_backend": "cache_dit"},
        {"lora_path": "/tmp/lora"},
        {"enable_sleep_mode": True},
        {"diffusion_compile_granularity": "full"},
        {"diffusion_attention_schedule": object()},
    ],
)
def test_unsupported_configuration_is_rejected(overrides) -> None:
    with pytest.raises(ValueError, match="Cosmos3 manual CUDA graphs require"):
        validate_manual_cuda_graphs(_config(**overrides), torch.device("cuda"), _offload())


def test_non_cuda_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cosmos3 manual CUDA graphs require"):
        validate_manual_cuda_graphs(_config(), torch.device("cpu"), _offload())


def test_offload_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cosmos3 manual CUDA graphs require"):
        validate_manual_cuda_graphs(
            _config(),
            torch.device("cuda"),
            _offload(OffloadStrategy.MODEL_LEVEL),
        )


def test_only_serialized_modelopt_fp8_quantization_is_accepted() -> None:
    fp8 = SimpleNamespace(get_name=lambda: "modelopt", is_checkpoint_fp8_serialized=True)
    validate_manual_cuda_graphs(_config(quantization_config=fp8), torch.device("cuda"), _offload())

    online_fp8 = SimpleNamespace(get_name=lambda: "modelopt", is_checkpoint_fp8_serialized=False)
    with pytest.raises(ValueError, match="Cosmos3 manual CUDA graphs require"):
        validate_manual_cuda_graphs(_config(quantization_config=online_fp8), torch.device("cuda"), _offload())

    other = SimpleNamespace(get_name=lambda: "fp8", is_checkpoint_fp8_serialized=True)
    with pytest.raises(ValueError, match="Cosmos3 manual CUDA graphs require"):
        validate_manual_cuda_graphs(_config(quantization_config=other), torch.device("cuda"), _offload())
