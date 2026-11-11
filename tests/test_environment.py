import torch
import torchvision


def test_torch_import():
    assert torch.__version__
    assert torchvision.__version__


def test_cpu_tensor_operations():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    result = a + b

    assert result.device.type == "cpu"
    assert torch.allclose(
        result,
        torch.tensor([5.0, 7.0, 9.0])
    )


def test_autograd():
    x = torch.tensor([2.0], requires_grad=True)

    loss = x.square().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.allclose(
        x.grad,
        torch.tensor([4.0])
    )


def test_resnet18_available():
    from torchvision.models import resnet18

    assert callable(resnet18)
