import torch

import pytest

from qcd_ml.nn.matrix_layers.bilinear import LGE_Bilinear, LGE_BilinearLM


@pytest.mark.benchmark(group="matrix_layers.bilinear")
def test_LGE_Bilinear(benchmark):
    n_input1 = 3
    n_input2 = 3
    n_output = 1
    layer = LGE_Bilinear(n_input1, n_input2, n_output)

    input1_features = torch.randn(n_input1, 8,8,8,16, 3,3, dtype=torch.cdouble)
    input2_features = torch.randn(n_input2, 8,8,8,16, 3,3, dtype=torch.cdouble)

    got = benchmark(layer.forward, input1_features, input2_features)


@pytest.mark.benchmark(group="matrix_layers.bilinear")
def test_LGE_BilinearLM(benchmark):
    n_input1 = 3
    n_input2 = 3
    n_output = 1
    layer = LGE_BilinearLM(n_input1, n_input2, n_output)
    layer_ref = LGE_Bilinear(n_input1, n_input2, n_output)
    layer_ref.weights.data = layer.weights.data

    input1_features = torch.randn(n_input1, 8,8,8,16, 3,3, dtype=torch.cdouble)
    input2_features = torch.randn(n_input2, 8,8,8,16, 3,3, dtype=torch.cdouble)

    got = benchmark(layer.forward, input1_features, input2_features)
    expect = layer_ref.forward(input1_features, input2_features)

    assert torch.allclose(expect, got)


@pytest.mark.benchmark(group="matrix_layers.bilinear.autograd")
def test_LGE_Bilinear_autograd(benchmark):
    n_input1 = 3
    n_input2 = 3
    n_output = 1
    layer = LGE_Bilinear(n_input1, n_input2, n_output)

    input1_features = torch.randn(n_input1, 8,8,8,16, 3,3, dtype=torch.cdouble, requires_grad=True)
    input2_features = torch.randn(n_input2, 8,8,8,16, 3,3, dtype=torch.cdouble, requires_grad=True)

    def dummy_training_step():
        out = layer.forward(input1_features, input2_features)
        loss = torch.abs(out.sum())
        loss.backward()
        return input1_features.grad, input2_features.grad, layer.weights.grad

    benchmark(dummy_training_step)


@pytest.mark.benchmark(group="matrix_layers.bilinear.autograd")
def test_LGE_BilinearLM_autograd(benchmark):
    n_input1 = 3
    n_input2 = 3
    n_output = 1
    layer = LGE_BilinearLM(n_input1, n_input2, n_output)
    layer_ref = LGE_Bilinear(n_input1, n_input2, n_output)
    layer_ref.weights.data = layer.weights.data

    input1_features = torch.randn(n_input1, 8,8,8,16, 3,3, dtype=torch.cdouble, requires_grad=True)
    input2_features = torch.randn(n_input2, 8,8,8,16, 3,3, dtype=torch.cdouble, requires_grad=True)
    layer.weights.grad = None
    input1_features.grad = None
    input2_features.grad = None

    def dummy_training_step():
        out = layer.forward(input1_features, input2_features)
        loss = torch.abs(out.sum())
        loss.backward()
        return input1_features.grad, input2_features.grad, layer.weights.grad

    got = dummy_training_step()
    # reset the gradients by setting them to zero prevents them from being
    # overwritten by the next backward pass
    layer.weights.grad = None
    input1_features.grad = None
    input2_features.grad = None
    benchmark(dummy_training_step)

    layer_ref.weights.grad = None
    input1_features.grad = None
    input2_features.grad = None
    out = layer_ref.forward(input1_features, input2_features)
    loss = torch.abs(out.sum())
    loss.backward()
    expect = (input1_features.grad, input2_features.grad, layer_ref.weights.grad)

    for i, (e, g) in enumerate(zip(expect, got)):
        print(i, torch.allclose(e, g))
        assert torch.allclose(e, g)
