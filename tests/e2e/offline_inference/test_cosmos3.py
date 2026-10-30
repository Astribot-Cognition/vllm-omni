# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Cosmos3 image-edit smoke test."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import DiffusionResponse, OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("VLLM_TEST_COSMOS3_MODEL", "nvidia/Cosmos3-Nano")
_OMNI_RUNNER_PARAM = (MODEL, None, {"model_config": {"guardrails": False}})

pytestmark = [
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
]


def _first_image(response: DiffusionResponse) -> Image.Image:
    assert response.images, "Cosmos3 edit returned no image"
    return response.images[0]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_image_edit_001(offline_client: OfflineOmniClient) -> None:
    """Default Cosmos3 image-edit smoke through the offline runner."""
    reference = Image.new("RGB", (640, 384), color=(40, 70, 110))
    response = offline_client.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "Move the yellow box to the left side of the table.",
            "modalities": ["image"],
            "multi_modal_data": {"image": reference},
            "sampling_params": OmniDiffusionSamplingParams(
                height=384,
                width=640,
                num_frames=5,
                num_inference_steps=2,
                guidance_scale=1.0,
                seed=42,
            ),
        }
    )
    result = _first_image(response)
    assert result.size == (640, 384)
