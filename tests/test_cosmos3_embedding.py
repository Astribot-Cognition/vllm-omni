# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for Cosmos3 pre-VAE-decode latent embeddings.

Covers the pipeline layout helper (final latent -> DiT patch view), the output
formatter routing into ``OmniRequestOutput.latents``, the embedding protocol,
and the ``/v1/embeddings`` handler with a mocked engine.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from pydantic import ValidationError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_post_process_func,
)
from vllm_omni.diffusion.output_formatter import (
    DiffusionPostprocessOutput,
    _format_single_prompt_output,
)
from vllm_omni.entrypoints.openai import serving_cosmos3_embedding as serving
from vllm_omni.entrypoints.openai.protocol.embeddings import (
    Cosmos3EmbeddingRequest,
    EmbeddingEncodingFormat,
)

pytestmark = [pytest.mark.cpu, pytest.mark.advanced_model]

LATENT_CHANNELS = 48
PATCH_SIZE = 2
LATENT_H = 24
LATENT_W = 40
PATCH_H = LATENT_H // PATCH_SIZE
PATCH_W = LATENT_W // PATCH_SIZE
PATCH_CHANNELS = PATCH_SIZE * PATCH_SIZE * LATENT_CHANNELS


class _TinyTransformer:
    """Minimal patchify surface shared by the pipeline helper."""

    latent_patch_size = PATCH_SIZE
    latent_channel_size = LATENT_CHANNELS

    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        padded_h = ((h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
        padded_w = ((w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
        return padded_h // PATCH_SIZE, padded_w // PATCH_SIZE, padded_h, padded_w

    def patchify(self, latents: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        batch = latents.shape[0]
        hp, wp, _, _ = self._pad_to_patch_size(h, w)
        x = latents.reshape(batch, LATENT_CHANNELS, t, hp, PATCH_SIZE, wp, PATCH_SIZE)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)
        return x.reshape(batch, t * hp * wp, PATCH_CHANNELS)


def _pipeline_with_tiny_transformer() -> Cosmos3OmniDiffusersPipeline:
    pipeline = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
    pipeline.transformer = _TinyTransformer()
    return pipeline


def test_latent_to_embedding_uses_patch_layout() -> None:
    latents = torch.randn(2, LATENT_CHANNELS, 1, LATENT_H, LATENT_W)
    pipeline = _pipeline_with_tiny_transformer()

    embedding = pipeline._latent_to_embedding(latents)

    assert embedding.shape == (2, PATCH_CHANNELS, 1, PATCH_H, PATCH_W)
    # Every patch channel block must equal the corresponding 2x2 latent tile.
    expected = (
        latents.reshape(2, LATENT_CHANNELS, 1, PATCH_H, PATCH_SIZE, PATCH_W, PATCH_SIZE)
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(2, 1, PATCH_H, PATCH_W, PATCH_CHANNELS)
        .permute(0, 4, 1, 2, 3)
    )
    assert torch.equal(embedding, expected)


def test_latent_to_embedding_round_trips_through_unpatchify() -> None:
    latents = torch.randn(1, LATENT_CHANNELS, 1, LATENT_H, LATENT_W)
    pipeline = _pipeline_with_tiny_transformer()
    embedding = pipeline._latent_to_embedding(latents)

    tokens = embedding.permute(0, 2, 3, 4, 1).reshape(1, PATCH_H * PATCH_W, PATCH_CHANNELS)
    restored = _TinyTransformer().patchify(latents, 1, LATENT_H, LATENT_W)
    assert torch.equal(tokens, restored)


def test_postprocess_passes_latent_payload_through() -> None:
    postprocess = get_cosmos3_post_process_func(
        SimpleNamespace(model_class_name="Cosmos3OmniDiffusersPipeline", model_config={})
    )
    tensor = torch.randn(1, PATCH_CHANNELS, 1, PATCH_H, PATCH_W)
    output = {"latent": tensor}
    sampling_params = SimpleNamespace(output_type="latent")

    assert postprocess(output, sampling_params=sampling_params) is output


def test_wants_latent_output_reads_output_type() -> None:
    assert Cosmos3OmniDiffusersPipeline._wants_latent_output(SimpleNamespace(output_type="latent"))
    assert not Cosmos3OmniDiffusersPipeline._wants_latent_output(SimpleNamespace(output_type="np"))
    assert not Cosmos3OmniDiffusersPipeline._wants_latent_output(SimpleNamespace())


def test_formatter_routes_latent_payload_to_latents_field() -> None:
    tensor = torch.randn(1, PATCH_CHANNELS, 1, PATCH_H, PATCH_W)
    postprocess_output = DiffusionPostprocessOutput(outputs={"latent": tensor}, primary_key="latent")
    request = SimpleNamespace(
        request_id="emb-test",
        prompt="prompt",
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            resolution=None,
            width=640,
            height=384,
            num_inference_steps=35,
        ),
    )

    outputs = _format_single_prompt_output(
        request=request,
        prompt="prompt",
        diffusion_output=DiffusionOutput(),
        outputs=[tensor],
        metrics={},
        postprocess_output=postprocess_output,
        is_text_output=False,
        is_audio_output=False,
        final_output_type="image",
        audio_sample_rate=None,
    )

    assert len(outputs) == 1
    assert outputs[0].final_output_type == "latents"
    assert outputs[0].latents is tensor
    assert outputs[0].images == []


def test_embedding_request_defaults_and_validation() -> None:
    request = Cosmos3EmbeddingRequest(input="move the cup", image="data:image/png;base64,AAAA")
    assert request.encoding_format == EmbeddingEncodingFormat.FLOAT
    assert request.size is None
    assert request.num_inference_steps is None
    with pytest.raises(ValidationError):
        Cosmos3EmbeddingRequest(input="missing image")
    with pytest.raises(ValidationError):
        Cosmos3EmbeddingRequest(input="x", image="y", encoding_format="float16")


def _fake_raw_request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(args=None)))


def _cosmos3_od_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_class_name="Cosmos3OmniDiffusersPipeline",
        model_config={"latent_channel": LATENT_CHANNELS, "latent_patch_size": PATCH_SIZE},
    )


def _patch_serving_engine(monkeypatch, *, latents: torch.Tensor, od_config=None) -> None:
    monkeypatch.setattr(serving, "_get_engine_and_model", lambda request: (object(), "cosmos3-test", [object()]))
    monkeypatch.setattr(
        serving,
        "_get_diffusion_od_config",
        lambda engine_client, request: od_config or _cosmos3_od_config(),
    )
    monkeypatch.setattr(serving, "get_stage_type", lambda stage_config: "diffusion")

    async def _fake_load_images(inputs, normalize_rgb=True):
        return [object()]

    monkeypatch.setattr(serving, "_load_input_images", _fake_load_images)

    async def _fake_generate(**kwargs):
        return SimpleNamespace(latents=latents, multimodal_output={})

    monkeypatch.setattr(serving, "_generate_with_async_omni", _fake_generate)


async def test_handler_returns_float_embedding(monkeypatch) -> None:
    latents = torch.randn(1, PATCH_CHANNELS, 1, PATCH_H, PATCH_W)
    _patch_serving_engine(monkeypatch, latents=latents)
    request = Cosmos3EmbeddingRequest(input="prompt", image="data:image/png;base64,AAAA", seed=7)

    response = await serving._create_embedding(request, _fake_raw_request())

    embedding = response.data[0].embedding
    assert isinstance(embedding, list)
    assert len(embedding) == PATCH_CHANNELS * PATCH_H * PATCH_W
    assert response.metadata["shape"] == [PATCH_CHANNELS, 1, PATCH_H, PATCH_W]
    assert response.metadata["latent_channels"] == LATENT_CHANNELS
    assert response.metadata["latent_grid"] == [LATENT_H, LATENT_W]
    assert response.metadata["patch_grid"] == [PATCH_H, PATCH_W]
    assert response.metadata["num_inference_steps"] == 35
    assert response.metadata["guidance_scale"] == 2.5
    assert response.metadata["seed"] == 7


async def test_handler_returns_base64_embedding(monkeypatch) -> None:
    latents = torch.randn(1, PATCH_CHANNELS, 1, PATCH_H, PATCH_W)
    _patch_serving_engine(monkeypatch, latents=latents)
    request = Cosmos3EmbeddingRequest(
        input="prompt",
        image="data:image/png;base64,AAAA",
        encoding_format=EmbeddingEncodingFormat.BASE64,
    )

    response = await serving._create_embedding(request, _fake_raw_request())

    encoded = response.data[0].embedding
    assert isinstance(encoded, str)
    decoded = base64.b64decode(encoded)
    assert len(decoded) == torch.tensor(response.metadata["shape"]).prod().item() * 4


async def test_handler_rejects_non_cosmos3_model(monkeypatch) -> None:
    latents = torch.randn(1, PATCH_CHANNELS, 1, PATCH_H, PATCH_W)
    od_config = SimpleNamespace(model_class_name="OtherPipeline", model_config={})
    _patch_serving_engine(monkeypatch, latents=latents, od_config=od_config)
    request = Cosmos3EmbeddingRequest(input="prompt", image="data:image/png;base64,AAAA")

    with pytest.raises(HTTPException) as excinfo:
        await serving._create_embedding(request, _fake_raw_request())
    assert excinfo.value.status_code == 400


async def test_handler_rejects_non_native_size(monkeypatch) -> None:
    _patch_serving_engine(monkeypatch, latents=torch.randn(1, PATCH_CHANNELS, 1, PATCH_H, PATCH_W))
    request = Cosmos3EmbeddingRequest(input="prompt", image="data:image/png;base64,AAAA", size="512x512")

    with pytest.raises(HTTPException) as excinfo:
        await serving._create_embedding(request, _fake_raw_request())
    assert excinfo.value.status_code == 400


def _write_export(root: Path, *, width: int, height: int, spatial: int = 16, patch: int = 2) -> None:
    (root / "vae").mkdir(parents=True, exist_ok=True)
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "export.json").write_text(json.dumps({"width": width, "height": height}))
    (root / "vae" / "config.json").write_text(json.dumps({"scale_factor_spatial": spatial}))
    (root / "transformer" / "config.json").write_text(json.dumps({"latent_patch_size": patch}))


def test_export_canvas_pads_the_content_to_the_patch_grid(tmp_path) -> None:
    _write_export(tmp_path, width=640, height=360)

    assert serving._export_canvas(str(tmp_path)) == (640, 384)


def test_export_canvas_follows_the_recorded_content_size(tmp_path) -> None:
    aligned = tmp_path / "aligned"
    padded = tmp_path / "padded"
    _write_export(aligned, width=512, height=288)
    _write_export(padded, width=100, height=60)

    assert serving._export_canvas(str(aligned)) == (512, 288)
    assert serving._export_canvas(str(padded)) == (128, 64)


def test_export_canvas_is_none_without_geometry(tmp_path) -> None:
    assert serving._export_canvas(str(tmp_path)) is None
    assert serving._served_canvas(SimpleNamespace(model=str(tmp_path))) == serving.FALLBACK_CANVAS
    assert serving._served_canvas(SimpleNamespace()) == serving.FALLBACK_CANVAS


def test_served_canvas_reads_the_model_export(tmp_path) -> None:
    _write_export(tmp_path, width=320, height=180)

    assert serving._served_canvas(SimpleNamespace(model=str(tmp_path))) == (320, 192)
