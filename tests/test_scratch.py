"""From-scratch models: shapes, missing modalities, losses, a tiny training run, and the app's serving code."""
import numpy as np
import pytest
import torch

from src.scratch.models import METHODS, MultimodalRec, count_params
from src.scratch.train import info_nce, supcon

VOCAB = 50


@pytest.mark.parametrize("method", METHODS)
def test_every_method_handles_missing_modalities(method):
    torch.manual_seed(0)
    m = MultimodalRec(method, VOCAB).eval()
    x, t = torch.randn(4, 3, 64, 64), torch.randint(2, VOCAB, (4, 16))
    t[:, 8:] = 0
    both = m.encode(x if m.use_img else None, t if m.use_txt else None)["f"]
    assert torch.allclose(both.norm(dim=-1), torch.ones(4), atol=1e-4)
    if m.use_img and m.use_txt:  # fused models answer image-only and text-only queries too
        assert m.encode(x, None)["f"].shape == both.shape
        assert m.encode(None, t)["f"].shape == both.shape
    empty = torch.zeros(2, 16, dtype=torch.long)  # an empty title must not produce NaNs
    if m.use_txt:
        assert torch.isfinite(m.encode(None, empty)["f"]).all()
    assert count_params(m) < 10_000_000  # small enough for CPU inference


def test_no_pretrained_weights_are_used():
    """Two freshly built models with different seeds must differ: nothing is loaded from a checkpoint."""
    torch.manual_seed(0)
    a = MultimodalRec("gated", VOCAB)
    torch.manual_seed(1)
    b = MultimodalRec("gated", VOCAB)
    assert not torch.equal(a.img.stem[0].weight, b.img.stem[0].weight)


def test_losses():
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    assert supcon(z, y, 0.1) > 0
    same = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    good, bad = info_nce(same, same, torch.tensor(np.log(10.0))), info_nce(same, z, torch.tensor(np.log(10.0)))
    assert good < bad  # aligned pairs give a lower contrastive loss


def test_tiny_training_run_improves_over_chance():
    from src.scratch.data import SCRATCH_DIR, ScratchData
    from src.scratch.train import RunConfig, train_run
    if not (SCRATCH_DIR / "meta.csv").exists():
        pytest.skip("run python -m src.scratch.data --source local first")
    data = ScratchData()
    _, res = train_run(RunConfig("text", "AdamW", 1e-3, epochs=2, batch=64), data, verbose=False)
    assert res["test"]["i2i"] > 0.2 and np.isfinite(res["history"][-1]["train_loss"])


def test_serving_code():
    from src.scratch import serve
    if not serve.available():
        pytest.skip("no exported models")
    from PIL import Image
    img = Image.open(serve.catalog().iloc[0]["image_file"])
    for m in serve.export_info()["methods"]:
        q = serve.encode_query(m, image=img, text="red dress")
        idx, s = serve.top_k(serve.scores(m, q), 5)
        assert len(idx) == 5 and np.all(np.diff(s) <= 1e-6)
