import torch.nn
import tqdm
import numpy as np
import pandas as pd
from torch.functional import F


def loss_function(pred, data):
    # TODO: functionality to change loss function
    return F.mse_loss(pred, data)


def fit(
    model: torch.nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_steps=1000,
    progress=True,
):
    """
    Training loop for torch model.
    """

    losses = []
    for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):

        pred = model.forward()
        loss = loss_function(pred, data)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if loss.is_cuda:
            losses.append(loss.cpu().detach().numpy())
        else:
            losses.append(loss.detach().numpy())

        if progress:
            pbar.set_description(f"Cost: {losses[-1]:.4f}")

    return losses


def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def kfold_validation(
    model: torch.nn.Module,
    datasets: list,
    optimizer: torch.optim.Optimizer,
    n_steps: int = 1000,
):

    d = []
    for i, data in enumerate(datasets):
        fit(model, data=data, optimizer=optimizer, n_steps=n_steps)
        pred = model.forward()

        _train = to_numpy(loss_function(pred, data))
        _tests = []
        for k, data_k in enumerate(datasets):
            if i == k:
                continue
            _tests.append(to_numpy(loss_function(pred, data_k)))

        d.append(
            dict(
                train=_train,
                test_mean=np.mean(_tests),
                test_std=np.std(_tests),
            )
        )

    return pd.DataFrame(d)
