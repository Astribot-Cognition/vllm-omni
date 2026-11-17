# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI-compatible embedding protocol for Cosmos3 latent embeddings.

The standard OpenAI embeddings request carries text only. Cosmos3 embeddings are
produced by image-edit inference, so this request extends the standard fields
with a required ``image`` reference and a few diffusion sampling knobs. The
response follows the OpenAI shape so existing embedding clients can read
``data[i].embedding`` unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingEncodingFormat(str, Enum):
    """Wire encoding of the embedding payload."""

    FLOAT = "float"
    BASE64 = "base64"


class Cosmos3EmbeddingRequest(BaseModel):
    """One Cosmos3 image-edit embedding request."""

    # OpenAI embedding clients may send fields this endpoint does not use
    # (e.g. ``user`` or ``dimensions``); ignore them instead of failing.
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, description="Model to use; defaults to the served model.")
    input: str = Field(..., description="Text prompt conditioning the edit.")
    image: str = Field(
        ...,
        description=(
            "Reference image as a data URL (``data:image/png;base64,...``) or an http(s) URL. "
            "Exactly one image is accepted."
        ),
    )
    size: str | None = Field(
        default=None,
        description="Output canvas in WIDTHxHEIGHT form. Cosmos3 image edit only supports 640x384.",
    )
    num_inference_steps: int | None = Field(default=None, gt=0, description="Diffusion steps (default 35).")
    guidance_scale: float | None = Field(
        default=None, gt=0, description="Classifier-free guidance scale (default 2.5)."
    )
    negative_prompt: str | None = Field(default=None, description="Negative prompt for CFG.")
    seed: int | None = Field(default=None, description="Random seed; a random seed is used and returned when omitted.")
    encoding_format: EmbeddingEncodingFormat = Field(
        default=EmbeddingEncodingFormat.FLOAT,
        description=(
            "'float' returns a JSON float list; 'base64' returns C-order little-endian float32 bytes "
            "encoded with base64. The layout is described by response metadata.shape."
        ),
    )


class EmbeddingData(BaseModel):
    """One embedding entry, matching the OpenAI embeddings response item."""

    object: str = "embedding"
    index: int = Field(..., description="Index of the input this embedding corresponds to.")
    embedding: list[float] | str = Field(
        ...,
        description="Embedding values; a list of floats for 'float' and a base64 string for 'base64'.",
    )


class EmbeddingUsage(BaseModel):
    """Token accounting; diffusion requests report zeros because they have no token output."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class Cosmos3EmbeddingResponse(BaseModel):
    """OpenAI-compatible embeddings response with Cosmos3 latent metadata."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Latent layout and sampling metadata: shape, dtype, source_dtype, latent_space, "
            "latent_channels, latent_grid, patch_grid, patch_size, num_inference_steps, "
            "guidance_scale and seed."
        ),
    )
