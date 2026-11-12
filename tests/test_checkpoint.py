
import torch

from train import save_checkpoint, load_checkpoint
from optimizers import Adam


def test_checkpoint_restore(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=0.01)

    x = torch.tensor([[1.0, 2.0]])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    saved_weights = [
        p.detach().clone() for p in model.parameters()
    ]

    path = tmp_path / "checkpoint.pt"
    history = [{"epoch": 1, "train_loss": 0.5}]

    save_checkpoint(
        path, model, optimizer, epoch=1, history=history
    )

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    epoch, restored_history = load_checkpoint(
        path, model, optimizer
    )

    assert epoch == 1
    assert restored_history == history

    for p, expected in zip(
        model.parameters(), saved_weights
    ):
        assert torch.allclose(p, expected)

    assert optimizer.steps == [1, 1]
