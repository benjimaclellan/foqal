import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from foqal.model import ModelBase
from foqal.utils.io import IO
from foqal.fit import fit


class ClassicalProbabilityCausalModel(ModelBase):

    def __init__(self, num_settings: int, latent_dim: int, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_settings = num_settings
        self.latent_dim = latent_dim


class ClassicalCommonCause(ClassicalProbabilityCausalModel):

    def __init__(self, num_settings: int, latent_dim: int, **kwargs):
        super().__init__(num_settings, latent_dim, **kwargs)

        self.terms = {
            "P(X=0|SL)": dict(
                shape=(self.num_settings, self.latent_dim),
                bounds=(0, 1),
            ),
            "P(Y=0|TL)": dict(
                shape=(self.num_settings, self.latent_dim),
                bounds=(0, 1),
            ),
            "P(L)": dict(
                shape=(self.latent_dim,),
                bounds=[0, 1],
            ),
        }

        self.initialize_params()
        return

    def forward(self):
        self.clip_params()
        t = {
            "P(X|SL)": torch.stack(
                [self.params["P(X=0|SL)"], 1.0 - self.params["P(X=0|SL)"]], dim=0
            ),
            "P(Y|TL)": torch.stack(
                [self.params["P(Y=0|TL)"], 1.0 - self.params["P(Y=0|TL)"]], dim=0
            ),
            "P(L)": self.params["P(L)"] / torch.sum(self.params["P(L)"]),
        }

        pred = torch.sum(
            t["P(X|SL)"][:, None, :, None, :]
            * t["P(Y|TL)"][None, :, None, :, :]
            * t["P(L)"][None, None, None, None, :],
            dim=4,
        )

        return pred


class Superdeterminism(ClassicalProbabilityCausalModel):

    def __init__(self, num_settings: int, latent_dim: int, **kwargs):
        super().__init__(num_settings, latent_dim, **kwargs)

        self.terms = {
            "P(X=0|SL)": dict(
                shape=(num_settings, latent_dim),
                bounds=(0, 1),
            ),
            "P(Y=0|TL)": dict(
                shape=(num_settings, latent_dim),
                bounds=(0, 1),
            ),
            "P(S|L)": dict(
                shape=(num_settings, latent_dim),
                bounds=(0, 1),
            ),
            "P(L)": dict(
                shape=(latent_dim,),
                bounds=(0, 1),
            ),
        }

        self.initialize_params()
        return

    def forward(self):
        self.clip_params()
        t = {
            "P(X|SL)": torch.stack(
                [self.params["P(X=0|SL)"], 1.0 - self.params["P(X=0|SL)"]], dim=0
            ),
            "P(Y|TL)": torch.stack(
                [self.params["P(Y=0|TL)"], 1.0 - self.params["P(Y=0|TL)"]], dim=0
            ),
            "P(S|L)": self.params["P(S|L)"]
            / torch.sum(self.params["P(S|L)"], dim=0)[None, :],
            "P(L)": self.params["P(L)"] / torch.sum(self.params["P(L)"]),
        }

        num = torch.sum(
            t["P(X|SL)"][:, None, :, None, :]
            * t["P(Y|TL)"][None, :, None, :, :]
            * t["P(S|L)"][None, None, :, None, :]
            * t["P(L)"][None, None, None, None, :],
            dim=4,
        )

        denom = torch.sum(
            t["P(S|L)"][None, None, :, None, :] * t["P(L)"][None, None, None, None, :],
            dim=4,
        )

        pred = num / denom

        return pred


class Superluminal(ClassicalProbabilityCausalModel):

    def __init__(self, num_settings: int, latent_dim: int, **kwargs):
        super().__init__(num_settings, latent_dim, **kwargs)

        self.terms = {
            "P(X=0|STL)": dict(
                shape=(num_settings, num_settings, latent_dim),
                bounds=(0, 1),
            ),
            "P(Y=0|TL)": dict(
                shape=(num_settings, latent_dim),
                bounds=(0, 1),
            ),
            "P(L)": dict(
                shape=(latent_dim,),
                bounds=(0, 1),
            ),
        }

        self.initialize_params()
        return

    def forward(self):
        self.clip_params()
        t = {
            "P(X|STL)": torch.stack(
                [self.params["P(X=0|STL)"], 1.0 - self.params["P(X=0|STL)"]], dim=0
            ),
            "P(Y|TL)": torch.stack(
                [self.params["P(Y=0|TL)"], 1.0 - self.params["P(Y=0|TL)"]], dim=0
            ),
            "P(L)": self.params["P(L)"] / torch.sum(self.params["P(L)"]),
        }

        pred = torch.sum(
            t["P(X|STL)"][:, None, :, :, :]
            * t["P(Y|TL)"][None, :, None, :, :]
            * t["P(L)"][None, None, None, None, :],
            dim=4,
        )

        return pred


if __name__ == "__main__":
    use_device = True

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
