# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the shared CUDA-graph replay wrapper."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion import cuda_graph
from vllm_omni.diffusion.cuda_graph import GraphedForward, clone_static, copy_into, kwargs_signature


class _Doubler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *, x: torch.Tensor, scale: float) -> torch.Tensor:
        self.calls += 1
        return x * scale


def test_signature_keys_tensor_structure_and_scalar_values() -> None:
    # Tensor VALUES do not key a graph: replay refreshes the static buffers.
    a = kwargs_signature({"x": torch.zeros(2, 3), "scale": 2.0})
    b = kwargs_signature({"x": torch.ones(2, 3) * 7, "scale": 2.0})
    assert a == b

    # Shape, dtype and non-tensor values do key it, because they change the
    # captured code path rather than only the inputs.
    assert kwargs_signature({"x": torch.zeros(3, 2), "scale": 2.0}) != a
    assert kwargs_signature({"x": torch.zeros(2, 3, dtype=torch.float64), "scale": 2.0}) != a
    assert kwargs_signature({"x": torch.zeros(2, 3), "scale": 3.0}) != a


def test_clone_static_and_copy_into_track_the_tree() -> None:
    live = {"x": torch.arange(6.0).view(2, 3), "inner": {"pos": torch.tensor([1, 2])}, "n": 4}
    static = clone_static(live)
    assert static["x"].data_ptr() != live["x"].data_ptr()
    assert torch.equal(static["x"], live["x"])

    live["x"].fill_(7.0)
    live["inner"]["pos"].fill_(9)
    assert copy_into(static, live) == 2
    assert torch.equal(static["x"], torch.full((2, 3), 7.0))
    assert torch.equal(static["inner"]["pos"], torch.tensor([9, 9]))

    live["x"].fill_(3.0)
    assert copy_into(static, live, batched=True) == 2
    assert torch.equal(static["x"], torch.full((2, 3), 3.0))

    with pytest.raises(ValueError, match="shape or dtype"):
        copy_into(static, {**live, "x": torch.zeros(3, 2)})


def test_wrapper_disabled_on_cpu_runs_eager() -> None:
    module = _Doubler()
    graph = GraphedForward(module, device=torch.device("cpu"))
    assert graph.enabled is False

    out = graph(x=torch.ones(2), scale=3.0)
    assert torch.equal(out, torch.full((2,), 3.0))
    assert module.calls == 1
    assert graph.eager_calls == 1 and graph.replays == 0


def test_capture_warms_up_before_the_sync_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _Doubler()
    graph = GraphedForward(module, device=torch.device("cpu"))
    graph.enabled = True

    @contextmanager
    def probe(mode: str):
        assert module.calls == graph.warmup_iters
        yield

    monkeypatch.setattr(cuda_graph, "_sync_debug_mode", probe)
    monkeypatch.setattr(cuda_graph.current_omni_platform, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(graph, "_warmup_stream", nullcontext)
    monkeypatch.setattr(graph, "_capture_graph", lambda body: (SimpleNamespace(replay=lambda: None), body()))

    graph(x=torch.ones(2), scale=2.0)
    assert graph.enabled is True
    assert graph.captures == 1
    assert graph.replays == 1


def test_capture_failure_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _Doubler()
    graph = GraphedForward(module, device=torch.device("cpu"))
    graph.enabled = True

    monkeypatch.setattr(cuda_graph.current_omni_platform, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(graph, "_warmup_stream", nullcontext)

    def explode(body):
        raise RuntimeError("capture is not allowed here")

    monkeypatch.setattr(graph, "_capture_graph", explode)

    assert torch.equal(graph(x=torch.ones(2), scale=4.0), torch.full((2,), 4.0))
    assert graph.enabled is False
    assert graph.captures == 0
    assert graph.eager_calls >= 1
