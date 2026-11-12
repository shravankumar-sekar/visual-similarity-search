
import pytest
import torch

from optimizers import MomentumSGD, Adam, NAdam


def test_sgd_first_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MomentumSGD([p], lr=0.1, momentum=0.9)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    assert torch.allclose(p.detach(), torch.tensor([0.8]))


def test_sgd_momentum():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MomentumSGD([p], lr=0.1, momentum=0.9)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    p.grad = torch.tensor([4.0])
    optimizer.step()

    assert torch.allclose(p.detach(), torch.tensor([0.22]))


def test_sgd_zero_grad():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MomentumSGD([p])

    p.grad = torch.tensor([2.0])
    optimizer.zero_grad()

    assert p.grad is None


def test_sgd_missing_gradient():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MomentumSGD([p])

    optimizer.step()

    assert p.item() == 1.0


def test_sgd_invalid_parameters():
    p = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError):
        MomentumSGD([p], lr=-0.1)

    with pytest.raises(ValueError):
        MomentumSGD([p], momentum=1.5)


def test_sgd_state_restore():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MomentumSGD([p], lr=0.1)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    state = optimizer.state_dict()

    q = torch.nn.Parameter(p.detach().clone())
    restored = MomentumSGD([q])
    restored.load_state_dict(state)

    p.grad = torch.tensor([4.0])
    q.grad = torch.tensor([4.0])

    optimizer.step()
    restored.step()

    assert torch.allclose(p, q)

def test_adam_first_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([p], lr=0.1)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    assert torch.allclose(
        p.detach(), torch.tensor([0.9]), atol=1e-6
    )


def test_adam_second_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam(
        [p], lr=0.1, beta1=0.5, beta2=0.5
    )

    p.grad = torch.tensor([2.0])
    optimizer.step()

    p.grad = torch.tensor([4.0])
    optimizer.step()

    assert torch.allclose(
        p.detach(),
        torch.tensor([0.80377496]),
        atol=1e-6,
    )


def test_adam_missing_gradient():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([p])

    optimizer.step()

    assert p.item() == 1.0
    assert optimizer.steps == [0]


def test_adam_invalid_parameters():
    p = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError):
        Adam([p], lr=0)

    with pytest.raises(ValueError):
        Adam([p], beta1=1.0)

    with pytest.raises(ValueError):
        Adam([p], beta2=-0.1)

    with pytest.raises(ValueError):
        Adam([p], eps=0)


def test_adam_state_restore():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([p], lr=0.01)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    state = optimizer.state_dict()

    q = torch.nn.Parameter(p.detach().clone())
    restored = Adam([q])
    restored.load_state_dict(state)

    p.grad = torch.tensor([4.0])
    q.grad = torch.tensor([4.0])

    optimizer.step()
    restored.step()

    assert torch.allclose(p.detach(), q.detach())
    assert restored.steps == optimizer.steps


def test_nadam_first_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = NAdam([p], lr=0.1)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    assert torch.allclose(
        p.detach(), torch.tensor([0.81]), atol=1e-6
    )


def test_nadam_second_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = NAdam(
        [p], lr=0.1, beta1=0.5, beta2=0.5
    )

    p.grad = torch.tensor([2.0])
    optimizer.step()

    p.grad = torch.tensor([4.0])
    optimizer.step()

    expected = torch.tensor([0.7249074])

    assert torch.allclose(
        p.detach(), expected, atol=1e-6
    )


def test_nadam_missing_gradient():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = NAdam([p])

    optimizer.step()

    assert p.item() == 1.0
    assert optimizer.steps == [0]


def test_nadam_invalid_parameters():
    p = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError):
        NAdam([p], lr=0)

    with pytest.raises(ValueError):
        NAdam([p], beta1=1.0)

    with pytest.raises(ValueError):
        NAdam([p], beta2=-0.1)

    with pytest.raises(ValueError):
        NAdam([p], eps=0)


def test_nadam_state_restore():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = NAdam([p], lr=0.01)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    state = optimizer.state_dict()

    q = torch.nn.Parameter(p.detach().clone())
    restored = NAdam([q])
    restored.load_state_dict(state)

    p.grad = torch.tensor([4.0])
    q.grad = torch.tensor([4.0])

    optimizer.step()
    restored.step()

    assert torch.allclose(p.detach(), q.detach())
    assert restored.steps == optimizer.steps
