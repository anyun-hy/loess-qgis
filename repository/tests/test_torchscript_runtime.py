import torch

from torchscript_runtime import (
    _insert_contiguous_before_permuted_convolutions,
    _mps_graph_requirement,
    _prepare_frozen_mps_model,
    load_torchscript_model,
)


class _FrozenFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 14, kernel_size=1)

    def forward(self, image):
        return self.conv(image)


class _PermuteDepthwiseConvFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)

    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        return self.conv(image)


class _SwinWindowFixture(torch.nn.Module):
    def forward(self, image):
        return torch.roll(image, shifts=(1, 1), dims=(2, 3))


def test_frozen_torchscript_loads_directly_on_cpu(tmp_path):
    fixture = _FrozenFixture().eval()
    example = torch.zeros(1, 3, 512, 512)
    traced = torch.jit.trace(fixture, example, strict=True)
    frozen = torch.jit.freeze(traced.eval())
    path = tmp_path / "fixture.torchscript.pt"
    torch.jit.save(frozen, path)

    model, runtime = load_torchscript_model(path, "cpu")
    with torch.inference_mode():
        output = model(example)

    assert tuple(output.shape) == (1, 14, 512, 512)
    assert output.dtype == torch.float32
    assert runtime == {
        "mode": "direct",
        "device": "cpu",
        "tensor_constant_count": 0,
        "mps_constant_count": 0,
        "mps_cpu_bridge_count": 0,
        "mps_contiguous_bridge_count": 0,
    }


def test_mps_graph_fix_materializes_only_permuted_convolution_inputs():
    fixture = _PermuteDepthwiseConvFixture().eval()
    example = torch.randn(1, 16, 16, 3)
    traced = torch.jit.trace(fixture, example, strict=True)
    frozen = torch.jit.freeze(traced.eval())

    with torch.inference_mode():
        expected = frozen(example)

    inserted = _insert_contiguous_before_permuted_convolutions(frozen.forward.graph)
    frozen.forward.graph.lint()

    assert inserted == 1
    assert _insert_contiguous_before_permuted_convolutions(frozen.forward.graph) == 0
    convolution_inputs = [
        node.inputsAt(0).node().kind()
        for node in frozen.forward.graph.nodes()
        if node.kind() == "aten::_convolution"
    ]
    assert convolution_inputs == ["aten::contiguous"]

    with torch.inference_mode():
        actual = frozen(example)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mps_prepare_reports_contiguous_bridge_count_on_cpu_fixture():
    fixture = _PermuteDepthwiseConvFixture().eval()
    example = torch.randn(1, 16, 16, 3)
    frozen = torch.jit.freeze(torch.jit.trace(fixture, example, strict=True).eval())

    runtime = _prepare_frozen_mps_model(frozen, "cpu")

    assert runtime["mode"] == "mps_frozen_hybrid"
    assert runtime["device"] == "cpu"
    assert runtime["mps_contiguous_bridge_count"] == 1
    assert runtime["mps_cpu_bridge_count"] == 0


def test_swin_window_graph_requires_pytorch_27_only_on_mps_loader_path():
    fixture = _SwinWindowFixture().eval()
    example = torch.randn(1, 3, 16, 16)
    frozen = torch.jit.freeze(torch.jit.trace(fixture, example, strict=True).eval())

    compatible, message = _mps_graph_requirement(frozen.forward.graph, "2.5.1")

    assert compatible is False
    assert "PyTorch >=2.7" in message
    assert _mps_graph_requirement(frozen.forward.graph, "2.7.0") == (True, "")
