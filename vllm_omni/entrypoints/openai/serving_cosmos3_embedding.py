# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 pre-VAE-decode latent embeddings over ``POST /v1/embeddings``.

Cosmos3 image-edit inference normally decodes the final latent with the Wan VAE
and returns an image. This route runs the same inference with
``output_type="latent"``, skips the VAE decode, and returns the denoised latent
in its DiT patch layout (``[B, p*p*C, T, H/p, W/p]``, e.g.
``[1, 192, 1, 12, 20]`` for the native 640x384 edit canvas) as an
OpenAI-compatible embedding.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pybase64
import torch
from fastapi import APIRouter, HTTPException, Request
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config, _get_engine_and_model
from vllm_omni.entrypoints.openai.diffusion import (
    MAX_UINT32_SEED,
    _generate_with_async_omni,
    apply_stage_default_sampling_params,
)
from vllm_omni.entrypoints.openai.image_api_utils import parse_size
from vllm_omni.entrypoints.openai.images.helpers import _load_input_images
from vllm_omni.entrypoints.openai.protocol.embeddings import (
    Cosmos3EmbeddingRequest,
    Cosmos3EmbeddingResponse,
    EmbeddingData,
    EmbeddingEncodingFormat,
    EmbeddingUsage,
)
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

logger = init_logger(__name__)

router = APIRouter()

COSMOS3_MODEL_CLASS = "Cosmos3OmniDiffusersPipeline"
# Canvas the endpoint falls back to when the served export records no content
# size; LumoWorld exports record it, plain Cosmos3 checkpoints do not.
FALLBACK_CANVAS = (640, 384)
DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_GUIDANCE_SCALE = 2.5


def _read_json(path: Path) -> Any:
    """Read one exported config, or ``None`` when it is absent or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=8)
def _export_canvas(model_path: str) -> tuple[int, int] | None:
    """Derive the served canvas from an export, or ``None`` when it records none.

    ``export.json`` holds the content size the checkpoint was trained on; the
    canvas pads that content up to the latent patch grid, i.e. the VAE spatial
    factor times the DiT patch size. For the native export this yields
    640x384 from 640x360 content.
    """
    root = Path(model_path)
    manifest = _read_json(root / "export.json")
    if not isinstance(manifest, Mapping):
        return None
    content_width = manifest.get("width")
    content_height = manifest.get("height")
    vae_config = _read_json(root / "vae" / "config.json")
    transformer_config = _read_json(root / "transformer" / "config.json")
    if not isinstance(content_width, int) or not isinstance(content_height, int):
        return None
    if not isinstance(vae_config, Mapping) or not isinstance(transformer_config, Mapping):
        return None
    spatial = vae_config.get("scale_factor_spatial")
    patch = transformer_config.get("latent_patch_size")
    if not isinstance(spatial, int) or not isinstance(patch, int) or spatial <= 0 or patch <= 0:
        return None
    alignment = spatial * patch
    canvas = (-(-content_width // alignment) * alignment, -(-content_height // alignment) * alignment)
    logger.info(
        "Cosmos3 embeddings canvas %dx%d derived from %s (content %dx%d, alignment %d)",
        canvas[0],
        canvas[1],
        root / "export.json",
        content_width,
        content_height,
        alignment,
    )
    return canvas


def _served_canvas(od_config: Any) -> tuple[int, int]:
    """Canvas this server accepts for embeddings requests."""
    model_path = getattr(od_config, "model", None)
    if isinstance(model_path, str) and model_path:
        canvas = _export_canvas(model_path)
        if canvas is not None:
            return canvas
    return FALLBACK_CANVAS


def _require_cosmos3(engine_client: Any, raw_request: Request) -> Any:
    """Return the diffusion od_config, or fail when the served model is not Cosmos3."""
    od_config = _get_diffusion_od_config(raw_request, engine_client)
    model_class_name = getattr(od_config, "model_class_name", None)
    if model_class_name != COSMOS3_MODEL_CLASS:
        raise HTTPException(
            status_code=400,
            detail=(
                "The embeddings endpoint returns Cosmos3 pre-VAE-decode latents and is only available for "
                f"{COSMOS3_MODEL_CLASS} servers; the served model is {model_class_name!r}."
            ),
        )
    return od_config


def _native_size(request: Cosmos3EmbeddingRequest, canvas: tuple[int, int]) -> tuple[int, int]:
    """Resolve the requested canvas, accepting only the served Cosmos3 edit size."""
    if request.size is None:
        return canvas
    try:
        width, height = parse_size(request.size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid size {request.size!r}: {exc}") from exc
    if (width, height) != canvas:
        raise HTTPException(
            status_code=400,
            detail=f"Cosmos3 image-edit inference only supports the native {canvas[0]}x{canvas[1]} canvas.",
        )
    return width, height


def _latent_tensor(result: Any) -> torch.Tensor:
    """Extract the latent tensor from one engine request output."""
    latent = getattr(result, "latents", None)
    if latent is None:
        multimodal_output = getattr(result, "multimodal_output", None)
        if isinstance(multimodal_output, Mapping):
            latent = multimodal_output.get("latent")
    if latent is None:
        raise HTTPException(status_code=500, detail="Cosmos3 generation finished without a latent embedding.")
    if not isinstance(latent, torch.Tensor):
        try:
            latent = torch.as_tensor(latent)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Unsupported latent output type: {type(latent)}") from exc
    if latent.ndim != 5:
        raise HTTPException(
            status_code=500,
            detail=f"Cosmos3 latent embedding must be 5-D [B, C, T, H, W], got shape {tuple(latent.shape)}.",
        )
    if latent.shape[0] != 1:
        raise HTTPException(
            status_code=500,
            detail=f"Cosmos3 embedding expects a single sample, got batch size {int(latent.shape[0])}.",
        )
    return latent[0]


def _embedding_metadata(
    od_config: Any,
    latent: torch.Tensor,
    *,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> dict[str, Any]:
    """Describe the latent layout and the sampling parameters that produced it."""
    model_config = getattr(od_config, "model_config", None) or {}
    if not isinstance(model_config, Mapping):
        model_config = {}
    patch_size = int(model_config.get("latent_patch_size", 2))
    shape = [int(dim) for dim in latent.shape]
    patch_channels, latent_frames, patch_h, patch_w = shape
    latent_channels = patch_channels // (patch_size * patch_size)
    return {
        "shape": shape,
        "dtype": "float32",
        "source_dtype": str(latent.dtype).replace("torch.", ""),
        "latent_space": "normalized_dit",
        "latent_channels": latent_channels,
        "latent_grid": [patch_h * patch_size, patch_w * patch_size],
        "patch_grid": [patch_h, patch_w],
        "patch_size": patch_size,
        "num_latent_frames": latent_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }


async def _create_embedding(
    request: Cosmos3EmbeddingRequest,
    raw_request: Request,
) -> Cosmos3EmbeddingResponse:
    started = time.perf_counter()
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)
    if request.model is not None and request.model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Model mismatch: request specifies {request.model!r} but server is running {model_name!r}.",
        )
    od_config = _require_cosmos3(engine_client, raw_request)

    width, height = _native_size(request, _served_canvas(od_config))
    try:
        pil_images = await _load_input_images([request.image])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid reference image: {exc}") from exc

    image_stage_ids = [index for index, config in enumerate(stage_configs) if get_stage_type(config) == "diffusion"]
    if not image_stage_ids:
        raise HTTPException(status_code=503, detail="No diffusion stage found in the multi-stage pipeline.")
    image_stage_id = image_stage_ids[0]

    omni_prompt = OmniTextPrompt(prompt=request.input, modalities=["image"])
    omni_prompt["multi_modal_data"] = {"image": pil_images}
    if request.negative_prompt is not None:
        omni_prompt["negative_prompt"] = request.negative_prompt
    omni_prompt["mm_processor_kwargs"] = {"target_h": height, "target_w": width}
    omni_prompt["height"] = height
    omni_prompt["width"] = width

    gen_params = OmniDiffusionSamplingParams()
    app_state_args = getattr(raw_request.app.state, "args", None)
    default_sampling_params = getattr(app_state_args, "default_sampling_params", None)
    apply_stage_default_sampling_params(default_sampling_params, gen_params, str(image_stage_id))
    # The joint checkpoint was trained on the bare world-model prompt; the
    # image-edit endpoint's default system prompt would add conditioning that
    # dataset feature extraction must not include.
    gen_params.extra_args = dict(gen_params.extra_args or {})
    gen_params.extra_args.setdefault("use_system_prompt", False)

    gen_params.width = width
    gen_params.height = height
    gen_params.width_not_provided = False
    gen_params.height_not_provided = False
    gen_params.num_outputs_per_prompt = 1
    gen_params.num_inference_steps = request.num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS
    gen_params.guidance_scale = request.guidance_scale or DEFAULT_GUIDANCE_SCALE
    seed = request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
    gen_params.seed = seed
    # Ask the Cosmos3 pipeline for the latent instead of a decoded image.
    gen_params.output_type = "latent"

    request_id = f"emb-{random_uuid()}"
    result = await _generate_with_async_omni(
        engine_client=engine_client,
        gen_params=gen_params,
        stage_configs=stage_configs,
        prompt=omni_prompt,
        request_id=request_id,
    )

    source_latent = _latent_tensor(result)
    metadata = _embedding_metadata(
        od_config,
        source_latent,
        num_inference_steps=int(gen_params.num_inference_steps),
        guidance_scale=float(gen_params.guidance_scale),
        seed=int(seed),
    )
    payload = source_latent.detach().to(torch.float32).contiguous().cpu().numpy()
    if request.encoding_format == EmbeddingEncodingFormat.BASE64:
        encoded: list[float] | str = pybase64.b64encode(payload.tobytes()).decode("ascii")
    else:
        encoded = payload.reshape(-1).tolist()

    logger.info(
        "Cosmos3 embedding: model=%s shape=%s steps=%d guidance=%.2f seed=%d latency=%.2fs",
        model_name,
        metadata["shape"],
        metadata["num_inference_steps"],
        metadata["guidance_scale"],
        metadata["seed"],
        time.perf_counter() - started,
    )
    return Cosmos3EmbeddingResponse(
        data=[EmbeddingData(index=0, embedding=encoded)],
        model=model_name,
        usage=EmbeddingUsage(),
        metadata=metadata,
    )


@router.post(
    "/v1/embeddings",
    response_model=Cosmos3EmbeddingResponse,
    responses={
        400: {"description": "Invalid request or unsupported model."},
        503: {"description": "Engine not initialized with a diffusion stage."},
    },
)
async def create_cosmos3_embedding(request: Cosmos3EmbeddingRequest, raw_request: Request) -> Cosmos3EmbeddingResponse:
    """Return the Cosmos3 pre-VAE-decode latent for prompt + reference-image inference."""
    try:
        return await _create_embedding(request, raw_request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Cosmos3 embedding request failed")
        raise HTTPException(status_code=500, detail=f"Cosmos3 embedding failed: {exc}") from exc
