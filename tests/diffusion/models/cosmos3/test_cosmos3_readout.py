# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the Cosmos3 readout heads.

The heads decode boxes/waypoints from LumoWorld's joint checkpoint; these tests
cover the parts that do not need a checkpoint: the exported recipe contract, the
token layout the heads assume, and the response serialization.
"""

from __future__ import annotations

import json

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.readout import (
    ReadoutConfig,
    ReadoutHeads,
    readout_config_path,
    readout_payload,
    token_layout,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def tiny_config(**overrides) -> ReadoutConfig:
    """Smallest recipe that keeps the two-query contract."""
    raw = {
        "revision": "cosmos-native-v3-place-pick-two-query-sigmoid-v3",
        "features": "both",
        "hidden_dim": 16,
        "layers": 1,
        "attention_heads": 2,
        "box_slots": 2,
        "waypoint_steps": 3,
        "box_presence": True,
        "queries_per_task": 2,
    }
    raw.update(overrides)
    return ReadoutConfig.from_dict(raw)


def test_config_round_trips_through_json(tmp_path) -> None:
    config = tiny_config()
    path = tmp_path / "readout" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({**config.__dict__, "unknown_key": 1}))

    loaded = ReadoutConfig.from_path(path)

    assert loaded == config
    assert loaded.box_outputs == 5
    assert tiny_config(box_presence=False).box_outputs == 4


def test_config_rejects_unknown_revision() -> None:
    with pytest.raises(ValueError, match="unsupported readout revision"):
        tiny_config(revision="cosmos-readout-does-not-exist")


def test_config_rejects_unknown_feature_selection() -> None:
    with pytest.raises(ValueError, match="unsupported readout feature selection"):
        tiny_config(features="vision")


def test_config_rejects_missing_contract_keys() -> None:
    with pytest.raises(ValueError, match="not a supported head contract"):
        ReadoutConfig.from_dict({"revision": "cosmos-native-v3-place-pick-two-query-sigmoid-v3"})


def test_readout_config_path_points_at_the_exported_component(tmp_path) -> None:
    assert readout_config_path(tmp_path) == tmp_path / "readout" / "config.json"


@pytest.mark.parametrize("features,expected_tokens", [("both", 20), ("und", 8), ("gen", 12)])
def test_token_layout_matches_the_edit_pack(features: str, expected_tokens: int) -> None:
    vision_grid, text_len = (2, 3), 2
    config = tiny_config(features=features)
    segments, xy = token_layout(vision_grid, text_len, torch.device("cpu"))

    assert segments.tolist() == [0] * 6 + [1] * text_len + [2] * 6 + [3] * 6
    assert xy.shape == (segments.numel(), 2)
    # Text rows carry no position; vision rows sit at patch centres.
    assert torch.equal(xy[6 : 6 + text_len], torch.zeros(text_len, 2))
    expected_x = (torch.arange(3, dtype=torch.float32) + 0.5) / 3
    expected_y = (torch.arange(2, dtype=torch.float32) + 0.5) / 2
    torch.testing.assert_close(xy[0], torch.stack([expected_x[0], expected_y[0]]), rtol=0, atol=0)

    keep = torch.isin(segments, segments.new_tensor({"und": (0, 1), "gen": (2, 3), "both": (0, 1, 2, 3)}[features]))
    assert int(keep.sum()) == expected_tokens


def test_decode_readouts_shapes_and_ranges() -> None:
    config = tiny_config()
    heads = ReadoutHeads(config, token_dim=8).eval()
    vision_grid, text_len = (2, 3), 2
    und = torch.randn(1, 6 + text_len, 8)
    gen = torch.randn(1, 12, 8)

    boxes, waypoints = heads.decode(
        und_hidden=und,
        gen_hidden=gen,
        vision_grid=vision_grid,
        text_len=text_len,
    )

    assert boxes.shape == (1, 2, 5)
    assert waypoints.shape == (1, 2, config.waypoint_steps, 2)
    assert waypoints.dtype == torch.float32
    assert torch.all((waypoints > 0) & (waypoints < 1))


def test_decode_readouts_rejects_a_non_edit_generation_block() -> None:
    config = tiny_config()
    heads = ReadoutHeads(config, token_dim=8).eval()

    with pytest.raises(ValueError, match="hidden tokens"):
        heads.decode(
            und_hidden=torch.randn(1, 8, 8),
            gen_hidden=torch.randn(1, 6, 8),  # one vision frame, not two
            vision_grid=(2, 3),
            text_len=2,
        )


def test_payload_clips_coordinates_and_scores_presence() -> None:
    config = tiny_config()
    boxes = torch.tensor([[[0.0, -0.5, 0.25, 1.5, 1.25], [10.0, 0.2, 0.3, 0.4, 0.5]]])
    waypoints = torch.tensor([[[[0.25, 0.5]], [[0.75, 0.125]]]])

    payload = readout_payload(config, boxes, waypoints)

    assert payload["boxes"]["place"] == {"xyxy": [0.0, 0.25, 1.0, 1.0], "score": 0.5}
    assert payload["boxes"]["pick"]["xyxy"] == [0.2, 0.3, 0.4, 0.5]
    assert payload["waypoints"]["left"] == [[0.25, 0.5]]
    assert payload["coordinate_space"] == "normalized_content_xy"


def test_payload_omits_scores_without_presence_logits() -> None:
    config = tiny_config(box_presence=False)
    boxes = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])

    payload = readout_payload(config, boxes, torch.zeros(1, 2, config.waypoint_steps, 2))

    assert "score" not in payload["boxes"]["place"]
    assert payload["boxes"]["place"]["xyxy"] == [0.1, 0.2, 0.3, 0.4]


def test_payload_serves_one_sample_per_request() -> None:
    config = tiny_config()
    with pytest.raises(ValueError, match="one sample per request"):
        readout_payload(config, torch.zeros(2, 2, 5), torch.zeros(2, 2, config.waypoint_steps, 2))
