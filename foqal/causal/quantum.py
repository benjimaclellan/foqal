import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import numpy as np

from foqal.model import ModelBase
from foqal.utils.io import IO
from foqal.fit import fit


class QuantumCommonCause(ModelBase):
    """
    The quantum common cause model is a parametrically radical, but structurally conservative model.

    :math:`P_{XY|ST} = \\text{Tr} \\left[ (E_{X|S} \\otimes E_{Y|T}) \\ \\rho_{AB}  \\right]`

    """

    def __init__(self, n_settings: int, latent_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.n_settings = n_settings

        if latent_dim > 4:
            raise UserWarning(
                "Are you sure you want a quantum causal model latent dimension greater than 4?"
            )
        if latent_dim > 8:
            raise RuntimeError(
                "A quantum causal model with latent dimension greater than 8 is not supported yet. "
            )

        self.latent_dim = latent_dim

        self.terms = {
            "E(X=0|S)": dict(
                shape=(self.n_settings, self.latent_dim, self.latent_dim),
                bounds=(-1, 1),
            ),
            "E(Y=0|T)": dict(
                shape=(self.n_settings, self.latent_dim, self.latent_dim),
                bounds=(-1, 1),
            ),
            "rho(AB)": dict(
                shape=(1, self.latent_dim**2, self.latent_dim**2),
                bounds=(-1, 1),
            ),
        }

        self.initialize_params()
        self.identity = nn.Parameter(
            torch.eye(self.latent_dim)[None, :, :], requires_grad=False
        )
        return

    @staticmethod
    def density_operators(param):
        # parameterize the T-matrix (dims 1,2) for each setting (dim 0)
        t_re = torch.tril(param, diagonal=0)
        t_im = torch.transpose(torch.triu(param, diagonal=1), -2, -1)
        t = t_re + 1j * t_im

        # generate a positive, semi-definite, hermitian operator with unit trace for each setting
        tt = torch.conj(torch.transpose(t, -2, -1)) @ t
        norm = torch.einsum("ijj", tt)[:, None, None]
        rho = tt / norm  # normalize by the trace along dims (1,2)
        return rho

    def forward(self):
        self.clip_params()

        ex = self.density_operators(self.params["E(X=0|S)"])
        ey = self.density_operators(self.params["E(Y=0|T)"])

        t = {
            "E(X|S)": torch.stack([ex, self.identity - ex], dim=0),
            "E(Y|T)": torch.stack([ey, self.identity - ey], dim=0),
            "rho(AB)": self.density_operators(self.params["rho(AB)"]),
        }

        out_shape = [
            2,
            2,
            self.n_settings,
            self.n_settings,
            self.latent_dim**2,
            self.latent_dim**2,
        ]

        # Kronecker product of all measurement outcomes, XY
        e = torch.reshape(
            torch.einsum("xsik,ytjl->xystijkl", t["E(X|S)"], t["E(Y|T)"]),
            shape=out_shape,
        )

        # matrix multiplication of [E(X|S) \otimes E(Y|T)] @ rho(AB)
        out = torch.einsum("xystij,qjk->xystik", e, t["rho(AB)"])

        pred = torch.einsum("xystii->xyst", out)
        return torch.real(pred)


if __name__ == "__main__":
    use_device = True

    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    device = torch.cuda.current_device()

    io = IO.directory(
        folder="entangled-state-data", include_date=False, include_id=False
    )

    run = 0
    m = 70
    p = 0.8
    latent_dim = 2

    data = torch.Tensor(io.load_np_array(filename=f"m={m}_p={p}_run{run}.npy"))
    if use_device:
        data = data.to(device)

    training_curves = {}

    model = QuantumCommonCause(n_settings=m, latent_dim=latent_dim)

    if use_device:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

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

    pred = model.forward()
    print(torch.sum(pred, (0, 1)))
