import pytest

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline


def test_non_edit_batch_is_rejected_with_clear_error() -> None:
    with pytest.raises(ValueError, match="image-edit requests only"):
        Cosmos3OmniDiffusersPipeline._validate_edit_batch_prompts(["a text-only prompt"])


def test_edit_without_reference_image_is_rejected() -> None:
    prompt = {"modalities": ["image"], "prompt": "edit", "additional_information": {}}
    with pytest.raises(ValueError, match="preprocessed reference image"):
        Cosmos3OmniDiffusersPipeline._validate_edit_batch_prompts([prompt])


def test_edit_batch_prompts_are_accepted() -> None:
    prompt = {"modalities": ["image"], "prompt": "edit", "additional_information": {"preprocessed_image": object()}}
    Cosmos3OmniDiffusersPipeline._validate_edit_batch_prompts([prompt, prompt])
