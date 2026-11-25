"""Two-query readout heads: place/pick boxes and left/right waypoints.

The heads are trained by LumoWorld's joint AlignVAE world model. They decode the
final hidden states of the two Cosmos3 towers: the understanding tower supplies
the source-vision and text segments, the generation tower the clean-source and
noisy-target segments. Coordinates are absolute and normalized to the image
content, matching the training objective; boxes carry a presence logit that
becomes a score after sigmoid, waypoints are stored as FP32 sigmoid.

Layout of ``cat(und_hidden, gen_hidden)`` along the sequence axis:

    0 source causal (vision) | 1 text | 2 source clean (vision) | 3 target noisy (vision)

Only the head computation lives here. Losses, labels and dataset-side filtering
stay in LumoWorld; serving consumes raw model outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

NATIVE_V3_READOUT_REVISION = "cosmos-native-v3-place-pick-two-query-sigmoid-v3"
LEGACY_READOUT_REVISION = "cosmos-place-pick-two-query-absolute-sigmoid-readout-v1"
SUPPORTED_READOUT_REVISIONS = (NATIVE_V3_READOUT_REVISION, LEGACY_READOUT_REVISION)

# Segment ids carried by each feature selection, mirroring the training recipe.
FEATURE_SEGMENTS = {"und": (0, 1), "gen": (2, 3), "both": (0, 1, 2, 3)}

# Entity slots are fixed by the training contract: box 0 is place, box 1 is
# pick; waypoint 0 is the left hand, waypoint 1 the right hand.
BOX_SLOTS = ("place", "pick")
WAYPOINT_SLOTS = ("left", "right")


@dataclass(frozen=True)
class ReadoutConfig:
    """Head contract recorded by the exporter next to the head weights."""

    revision: str
    features: str
    hidden_dim: int
    layers: int
    attention_heads: int
    box_slots: int
    waypoint_steps: int
    box_presence: bool
    queries_per_task: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReadoutConfig:
        try:
            config = cls(
                revision=str(raw["revision"]),
                features=str(raw["features"]),
                hidden_dim=int(raw["hidden_dim"]),
                layers=int(raw["layers"]),
                attention_heads=int(raw["attention_heads"]),
                box_slots=int(raw["box_slots"]),
                waypoint_steps=int(raw["waypoint_steps"]),
                box_presence=bool(raw["box_presence"]),
                queries_per_task=int(raw["queries_per_task"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("readout/config.json is not a supported head contract") from exc
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> ReadoutConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        if self.revision not in SUPPORTED_READOUT_REVISIONS:
            raise ValueError(f"unsupported readout revision: {self.revision}")
        if self.features not in FEATURE_SEGMENTS:
            raise ValueError(f"unsupported readout feature selection: {self.features}")
        if self.box_slots != len(BOX_SLOTS) or self.queries_per_task != 2:
            raise ValueError("readout heads require two entity queries over two box slots")
        if self.waypoint_steps < 1 or self.layers < 1 or self.attention_heads < 1:
            raise ValueError("readout head dimensions must be positive")

    @property
    def box_outputs(self) -> int:
        return 5 if self.box_presence else 4


class QueryHead(nn.Module):
    """Learned entity queries decode hidden-state tokens; one linear output per query."""

    def __init__(
        self,
        token_dim: int,
        queries: int,
        outputs: int,
        *,
        hidden_dim: int = 256,
        layers: int = 2,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        self.segment = nn.Embedding(4, hidden_dim)
        self.position = nn.Linear(2, hidden_dim)
        self.queries = nn.Parameter(torch.empty(queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            hidden_dim,
            attention_heads,
            hidden_dim * 4,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim, outputs)
        # Raw-coordinate heads start near the image centre.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.constant_(self.output.bias, 0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        segments: torch.Tensor,
        xy: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens.to(self.output.weight.dtype)
        xy = xy.to(tokens.dtype)
        memory = self.input_projection(tokens) + self.segment(segments) + self.position(xy)
        return self.output(
            self.decoder(self.queries.expand(len(tokens), -1, -1), memory, memory_key_padding_mask=padding_mask)
        )


class ReadoutHeads(nn.Module):
    """Container for the two heads, registered under one weight-loading prefix."""

    def __init__(self, config: ReadoutConfig, token_dim: int):
        super().__init__()
        size = {"hidden_dim": config.hidden_dim, "layers": config.layers, "attention_heads": config.attention_heads}
        self.config = config
        self.box_head = QueryHead(token_dim, config.queries_per_task, config.box_outputs, **size)
        self.waypoint_head = QueryHead(token_dim, config.queries_per_task, config.waypoint_steps * 2, **size)
        nn.init.zeros_(self.waypoint_head.output.bias)

    def decode(self, *, und_hidden, gen_hidden, vision_grid, text_len, padding_mask=None):
        return decode_readouts(
            self.config,
            self.box_head,
            self.waypoint_head,
            und_hidden=und_hidden,
            gen_hidden=gen_hidden,
            vision_grid=vision_grid,
            text_len=text_len,
            padding_mask=padding_mask,
        )


def token_layout(vision_grid: tuple[int, int], text_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment ids and patch-centre xy for ``cat(und_hidden, gen_hidden)``.

    Vision tokens are row-major patches of the padded latent canvas; text tokens
    sit between the two understanding segments and carry ``xy = 0``.
    """
    grid_h, grid_w = int(vision_grid[0]), int(vision_grid[1])
    vision_tokens = grid_h * grid_w
    text_len = int(text_len)
    counts = torch.tensor([vision_tokens, text_len, vision_tokens, vision_tokens], device=device)
    segments = torch.repeat_interleave(torch.arange(4, device=device), counts)
    y = (torch.arange(grid_h, device=device) + 0.5) / grid_h
    x = (torch.arange(grid_w, device=device) + 0.5) / grid_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack([xx, yy], -1).reshape(vision_tokens, 2)
    return segments, torch.cat([xy, xy.new_zeros(text_len, 2), xy, xy])


def decode_readouts(
    config: ReadoutConfig,
    box_head: QueryHead,
    waypoint_head: QueryHead,
    *,
    und_hidden: torch.Tensor,
    gen_hidden: torch.Tensor,
    vision_grid: tuple[int, int],
    text_len: int,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode raw box logits ``[B, slots, 5]`` and FP32-sigmoid waypoints ``[B, 2, k, 2]``."""
    tokens = torch.cat([und_hidden, gen_hidden], dim=1)
    batch = tokens.shape[0]
    segments, xy = token_layout(vision_grid, text_len, tokens.device)
    if tokens.shape[1] != segments.numel():
        raise ValueError(
            f"readout expects {segments.numel()} hidden tokens, got {tokens.shape[1]}; "
            "check the UND vision grid, text length and generation frame count"
        )
    keep = torch.isin(segments, segments.new_tensor(FEATURE_SEGMENTS[config.features]))
    tokens, segments, xy = tokens[:, keep], segments[keep], xy[keep]
    if padding_mask is not None:
        padding_mask = padding_mask[:, keep]
    boxes = box_head(tokens, segments, xy, padding_mask).reshape(batch, config.box_slots, config.box_outputs)
    waypoints = waypoint_head(tokens, segments, xy, padding_mask)
    waypoints = waypoints.reshape(batch, len(WAYPOINT_SLOTS), config.waypoint_steps, 2).float().sigmoid()
    return boxes, waypoints


def readout_payload(config: ReadoutConfig, boxes: torch.Tensor, waypoints: torch.Tensor) -> dict[str, Any]:
    """Serialize one sample of head outputs into the response body.

    Box coordinates are clipped to the closed image bounds, presence logits
    become scores, and waypoints are already normalized probabilities.
    """
    if boxes.shape[0] != 1 or waypoints.shape[0] != 1:
        raise ValueError("Cosmos3 readouts serve one sample per request")
    boxes = boxes[0].detach()
    waypoints = waypoints[0].detach()
    scores = boxes[..., 0].float().sigmoid() if config.box_presence else None
    coordinates = boxes[..., 1 if config.box_presence else 0 :].float().clamp(0, 1)
    payload: dict[str, Any] = {
        "revision": config.revision,
        "coordinate_space": "normalized_content_xy",
        "boxes": {},
        "waypoints": {},
    }
    for index, slot in enumerate(BOX_SLOTS[: boxes.shape[0]]):
        entry: dict[str, Any] = {"xyxy": [round(float(value), 6) for value in coordinates[index]]}
        if scores is not None:
            entry["score"] = round(float(scores[index]), 6)
        payload["boxes"][slot] = entry
    for index, slot in enumerate(WAYPOINT_SLOTS[: waypoints.shape[0]]):
        payload["waypoints"][slot] = [
            [round(float(value), 6) for value in point] for point in waypoints[index].cpu()
        ]
    return payload
