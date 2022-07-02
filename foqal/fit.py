import numpy as np
import time
import tqdm
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F


from foqal.utils.io import IO

from foqal.causal.classical import (
    ClassicalCommonCause,
    Superdeterminism,
    Superluminal,
)

use_device = True


def fit(model, data, optimizer, n_steps=1000):
    """
    Training loop for torch model.
    """

    losses = []
    for step in (pbar := tqdm.tqdm(range(n_steps))):

        pred = model.forward()
        loss = F.mse_loss(pred, data)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if use_device:
            losses.append(loss.cpu().detach().numpy())
        else:
            losses.append(loss.detach().numpy())

        pbar.set_description(f"Cost: {losses[-1]:.4f}")

    return losses


if __name__ == "__main__":

    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    device = torch.cuda.current_device()

    io = IO.create_new_save_folder(
        folder="entangled-state-data", include_date=False, include_uuid=False
    )

    run = 0
    m = 10
    p = 0.0
    latent_dim = 10

    data = torch.Tensor(io.load_np_array(filename=f"m={m}_p={p}_run{run}.npy"))
    if use_device:
        data = data.to(device)

    training_curves = {}

    for Model in [
        ClassicalCommonCause,
        Superdeterminism,
        Superluminal,
    ]:
        model = Model(num_settings=m, latent_dim=latent_dim)

        if use_device:
            model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        t0 = time.time()
        losses = fit(model, data, optimizer, n_steps=400)

        training_curves[model.__class__.__name__] = losses

        print(
            f"\n{model.__class__.__name__} | "
            f"\n\tTotal time: {time.time() - t0}| "
            f"\n\tTotal parameters: {sum(p.numel() for p in model.parameters())}"
            f"\n\tFinal loss: {losses[-1]}"
        )

        torch.cuda.empty_cache()

    fig, ax = plt.subplots(1, 1)
    for label, losses in training_curves.items():
        ax.plot(np.arange(len(losses)), np.log(losses), label=f"{label}")

    ax.legend()
    plt.show()
