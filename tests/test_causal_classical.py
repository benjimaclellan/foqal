import pytest
import torch

from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal


@pytest.mark.parametrize("n_settings", [10, 30, 50])
def test_causal_classical_total_probability(n_settings):
    for Model in [ClassicalCommonCause, Superdeterminism, Superluminal]:
        model = Model(n_settings=n_settings, latent_dim=n_settings)

        pred = model.forward()
        print(torch.all(torch.isclose(torch.sum(pred, (0, 1)), torch.tensor([1.0]))))
        assert torch.all(torch.isclose(torch.sum(pred, (0, 1)), torch.tensor([1.0])))
        assert pred.shape == torch.Size([2, 2, n_settings, n_settings])