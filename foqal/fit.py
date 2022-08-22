import torch.nn
import tqdm
import numpy as np
import pandas as pd


def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def fit(
    model: torch.nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss: torch.nn.modules.loss._Loss,
    n_steps=1000,
    progress=True,
):
    """
    One training loop for a model to fit to a single dataset.

    :param model:
    :param data:
    :param optimizer:
    :param loss:
    :param n_steps:
    :param progress:
    :return:
    """

    ls = []
    for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):

        pred = model.forward()

        if type(loss) is torch.nn.modules.KLDivLoss:
            pred = pred.log()

        l = loss(pred, data)

        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        ls.append(to_numpy(l))

        if progress:
            pbar.set_description(f"Cost: {ls[-1]:.10f}")

    return ls


def cross_validation(
    model: torch.nn.Module,
    datasets: list,
    optimizer: torch.optim.Optimizer,
    loss: torch.nn.modules.loss._Loss,
    n_steps: int = 1000,
):
    """
    Performs cross validation on a list of datasets.
    The model is trained on each dataset and tested on the remaining.
    This is repeated all k times.

    :param model: model object
    :param datasets: list of datasets
    :param optimizer: torch optimizer
    :param loss: torch loss function
    :param n_steps: number of steps to use for each fit
    :return:
    """
    d = []
    for i, data in enumerate(datasets):
        losses = fit(model, data=data, optimizer=optimizer, loss=loss, n_steps=n_steps)
        pred = model.forward()

        _train_loss = to_numpy(loss(pred, data))
        _test_losses = []
        for k, data_k in enumerate(datasets):
            if i == k:
                continue
            _test_losses.append(to_numpy(loss(pred, data_k)))

        d.append(
            dict(
                train=_train_loss,
                test_mean=np.mean(_test_losses),
                test_std=np.std(_test_losses),
                training_curve=losses,
            )
        )

    return d


def convert_array_to_mat(ar: np.ndarray, num_settings: int):
    """
    Converts from the nd-array data format to the matrix format
    Parameters
    ----------
    ar: 4-dimensional array representing the data frequency outcomes
    num_settings: number of settings for the data (required for proper indexing)

    Returns
    -------
    d: a 2-d numpy array in the 4 x m^2 matrix form (more human-readable)
    """
    d = np.zeros([num_settings * num_settings, 4])
    for x in (0, 1):
        for y in (0, 1):
            for s in range(num_settings):
                for t in range(num_settings):
                    d[s * num_settings + t, 2 * x + y] = ar[x, y, s, t]
    return d.T
