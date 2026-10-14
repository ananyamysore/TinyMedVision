import torch
from torch.utils.data import DataLoader, TensorDataset

from tinymedvision.models import build_model
from tinymedvision.training import load_checkpoint, save_checkpoint, seed_everything, train_model


def test_model_shapes_and_finite_probabilities():
    for name in ("compact_cnn", "depthwise_cnn", "residual_cnn"):
        model = build_model(name)
        logits = model(torch.rand(3, 1, 28, 28))
        assert logits.shape == (3, 2)
        assert torch.isfinite(torch.softmax(logits, 1)).all()


def test_train_checkpoint_roundtrip(tmp_path):
    seed_everything(4)
    images = torch.rand(8, 1, 16, 16)
    labels = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    model = build_model("compact_cnn", width=4)
    path = tmp_path / "model.pt"
    history = train_model(model, loader, loader, epochs=1, checkpoint=path)
    assert len(history) == 1 and path.is_file()
    expected = model(images).detach()
    restored = build_model("compact_cnn", width=4)
    metadata = load_checkpoint(restored, path)
    assert metadata["epoch"] == 1
    assert torch.allclose(expected, restored(images))
