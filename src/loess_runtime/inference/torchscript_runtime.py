"""Load formal frozen TorchScript artifacts on CPU, CUDA, or macOS MPS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F


_MPS_LIBRARY = None
_MPS_POOL_OP = "loess_deployment::adaptive_avg_pool2d_cpu_bridge"
_CONVOLUTION_OPS = {
    "aten::_convolution",
    "aten::conv2d",
    "aten::convolution",
}


def _walk_nodes(block) -> Iterator[Any]:
    for node in block.nodes():
        yield node
        for child in node.blocks():
            yield from _walk_nodes(child)


def _version_tuple(value: Any) -> tuple[int, ...]:
    parts = []
    for component in str(value).split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    return tuple(parts)


def _mps_graph_requirement(graph, torch_version: Any) -> tuple[bool, str]:
    has_swin_window_ops = any(node.kind() == "aten::roll" for node in _walk_nodes(graph))
    if has_swin_window_ops and _version_tuple(torch_version) < (2, 7):
        return (
            False,
            f"Swin MPS requires PyTorch >=2.7 in this deployment; current={torch_version}",
        )
    return True, ""


def _pool_cpu_bridge(input_tensor, output_size):
    pooled = F.adaptive_avg_pool2d(input_tensor.to("cpu"), output_size)
    return pooled.to(input_tensor.device)


def _ensure_mps_pool_bridge() -> None:
    global _MPS_LIBRARY
    if _MPS_LIBRARY is not None:
        return
    library = torch.library.Library("loess_deployment", "DEF")
    library.define("adaptive_avg_pool2d_cpu_bridge(Tensor input, int[] output_size) -> Tensor")
    library.impl("adaptive_avg_pool2d_cpu_bridge", _pool_cpu_bridge, "MPS")
    _MPS_LIBRARY = library


def _find_int_constant(graph, expected: int):
    for node in _walk_nodes(graph):
        if node.kind() != "prim::Constant" or not node.hasAttribute("value"):
            continue
        try:
            if node.output().type().kind() == "IntType" and node.i("value") == expected:
                return node.output()
        except (RuntimeError, TypeError):
            continue
    return None


def _insert_contiguous_before_permuted_convolutions(graph) -> int:
    """Materialize permuted convolution inputs for the MPS backend only."""

    memory_format = _find_int_constant(graph, 0)
    candidates = [
        node for node in list(_walk_nodes(graph))
        if node.kind() in _CONVOLUTION_OPS
        and node.inputsAt(0).node().kind() == "aten::permute"
    ]
    if candidates and memory_format is None:
        with graph.insert_point_guard(candidates[0]):
            memory_format = graph.insertConstant(0)

    inserted = 0
    for convolution in candidates:
        input_value = convolution.inputsAt(0)
        contiguous = graph.create(
            "aten::contiguous",
            [input_value, memory_format],
            1,
        )
        contiguous.output().setType(input_value.type())
        contiguous.insertBefore(convolution)
        convolution.replaceInput(0, contiguous.output())
        inserted += 1
    return inserted


def _prepare_frozen_mps_model(model, device: str) -> dict[str, Any]:
    """Move frozen weights to MPS and bridge backend-specific graph gaps.

    Task-one artifacts are frozen, so their weights live in graph constants and
    ``module.to('mps')`` cannot move them. Zero-dimensional constants stay on
    CPU because PyTorch permits CPU scalar operands and MPS cannot store the
    exported float64 scalar. UPerNet adaptive pooling is explicitly bridged
    through CPU because non-divisible pooling is unsupported by macOS MPS.
    """

    graph = model.forward.graph
    nodes = list(_walk_nodes(graph))
    tensor_constants = 0
    moved_constants = 0
    for node in nodes:
        if node.kind() != "prim::Constant" or not node.hasAttribute("value"):
            continue
        try:
            tensor = node.t("value")
        except (RuntimeError, TypeError):
            continue
        tensor_constants += 1
        if tensor.ndim == 0:
            continue
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise RuntimeError(
                f"MPS frozen constant must be float32, got {tensor.dtype} shape={tuple(tensor.shape)}"
            )
        if tensor.dtype == torch.float32:
            node.t_("value", tensor.to(device))
            moved_constants += 1

    contiguous_bridges = _insert_contiguous_before_permuted_convolutions(graph)

    pool_bridges = 0
    pool_nodes = [
        node for node in list(_walk_nodes(graph))
        if node.kind() == "aten::adaptive_avg_pool2d"
    ]
    if pool_nodes:
        _ensure_mps_pool_bridge()
    for node in pool_nodes:
        replacement = graph.create(_MPS_POOL_OP, list(node.inputs()), 1)
        replacement.output().setType(node.output().type())
        replacement.insertBefore(node)
        node.output().replaceAllUsesWith(replacement.output())
        node.destroy()
        pool_bridges += 1

    graph.lint()
    model.to(device)
    return {
        "mode": "mps_frozen_hybrid",
        "device": device,
        "tensor_constant_count": tensor_constants,
        "mps_constant_count": moved_constants,
        "mps_cpu_bridge_count": pool_bridges,
        "mps_contiguous_bridge_count": contiguous_bridges,
    }


def load_torchscript_model(path: str | Path, device: str):
    """Return an eval TorchScript model and auditable runtime metadata."""

    target = str(device)
    if target.startswith("mps"):
        model = torch.jit.load(str(path), map_location="cpu").eval()
        compatible, message = _mps_graph_requirement(model.forward.graph, torch.__version__)
        if not compatible:
            raise RuntimeError(message)
        return model, _prepare_frozen_mps_model(model, target)

    model = torch.jit.load(str(path), map_location=target).eval()
    return model, {
        "mode": "direct",
        "device": target,
        "tensor_constant_count": 0,
        "mps_constant_count": 0,
        "mps_cpu_bridge_count": 0,
        "mps_contiguous_bridge_count": 0,
    }
