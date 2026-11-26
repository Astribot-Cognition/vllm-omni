# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any, Literal

from PIL import Image

COSMOS3_EXTRA_BODY_PARAMS = frozenset(
    {
        "flow_shift",
        "max_sequence_length",
        "use_resolution_template",
        "use_duration_template",
        "negative_metadata_mode",
        "use_system_prompt",
        "system_prompt",
        "negative_prompt",
        "guardrails",
        "condition_frame_indexes_vision",
        "condition_video_keep",
        "generate_sound",
        "sound_gen",
        "sound_duration",
        "audio_duration",
        "action_mode",
        "action",
        "domain_name",
        "domain_id",
        "raw_action_dim",
        "action_chunk_size",
        "action_space",
        "action_fps",
        "image_height",
        "image_width",
        "history_length",
        "conditioning_fps",
        "resolution",
        "image_size",
        "use_state",
        "format_prompt_as_json",
        "observation",
        "robot_obs",
        "deterministic_seed",
        "session_id",
    }
)
COSMOS3_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "action",
        "raw_action_dim",
        "domain_id",
        "action_mode",
        "readouts",
    }
)


ReadoutCapability = Literal["present", "absent", "unknown"]


def cosmos3_readout_capability(od_config: Any) -> ReadoutCapability:
    """Whether the served Cosmos3 export ships the readout heads.

    ``unknown`` means the request cannot be judged here (another model class, or
    no on-disk model path), so callers must not treat it as absent.
    """
    if getattr(od_config, "model_class_name", None) != "Cosmos3OmniDiffusersPipeline":
        return "unknown"
    model_path = getattr(od_config, "model", None)
    if not model_path:
        return "unknown"
    from vllm_omni.diffusion.models.cosmos3.readout import readout_config_path

    return "present" if readout_config_path(model_path).is_file() else "absent"


def build_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Build a Cosmos3 edit request carrying an image modality."""
    del height, width
    if isinstance(input_image, list):
        if len(input_image) != 1:
            raise ValueError("Cosmos3 image edit supports a single reference image.")
        input_image = input_image[0]
    result: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
        "multi_modal_data": {"image": input_image},
    }
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result
