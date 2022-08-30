import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from foqal.model import ModelBase
from foqal.utils.io import IO
from foqal.fit import fit, to_numpy
from foqal.optim import KLDivLoss
from foqal.causal.quantum import QuantumCommonCause


class ClassicalProbabilityCausalModel(ModelBase):
    """
    Base model for all classical probabilistic (i.e., parametrically conservative) models.
    """

    def __init__(self, n_settings: int, latent_dim: int, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.n_settings = n_settings
        self.latent_dim = latent_dim
        # self.prob_bounds = (1e-14, 1.0)  # TODO: may be necessary for KL divergence to avoid instability
        self.prob_bounds = (0.0, 1.0)


class ClassicalCommonCause(ClassicalProbabilityCausalModel):
    """
    This model is both parameterically and structurally conservative,
    and can be defined as a classical local, hidden-variable model.

    The joint probability of binary measurement outcomes X and Y for settings S and T is,
    :math:`P_{XY|ST} = \\sum_{\\lambda \in \\Lambda} P_{X|S\\lambda} P_{Y|T\\lambda} P_{\\lambda}`

    Number of parameters:
    :math:`|\\Lambda | (2m + 1)`
    """

    def __init__(self, n_settings: int, latent_dim: int, **kwargs):
        """
        :param n_settings: number of measurement settings
        :param latent_dim: cardinality of the latent variable
        :param kwargs:
        """
        super().__init__(n_settings, latent_dim, **kwargs)

        self.terms = {
            "P(X=0|SL)": dict(
                shape=(self.n_settings, self.latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(Y=0|TL)": dict(
                shape=(self.n_settings, self.latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(L)": dict(
                shape=(self.latent_dim,),
                bounds=self.prob_bounds,
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
    """
    This model is parameterically conservative and structurally radical.

    The joint probability of binary measurement outcomes X and Y for settings S and T is,
    :math:`P_{XY|ST} = \\sum_{\\lambda \\in \\Lambda} P_{X|S\\lambda} P_{Y|T\\lambda} P_{S | \\lambda} P_{\\lambda} \\left( \\sum_{\\lambda \\in \\Lambda} P_{S|\\lambda'} P_{\\lambda} \\right)^{-1}`

    Number of parameters:
    :math:`|\\Lambda | (3m + 1)`
    """

    def __init__(self, n_settings: int, latent_dim: int, **kwargs):
        super().__init__(n_settings, latent_dim, **kwargs)

        self.terms = {
            "P(X=0|SL)": dict(
                shape=(n_settings, latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(Y=0|TL)": dict(
                shape=(n_settings, latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(S|L)": dict(
                shape=(n_settings, latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(L)": dict(
                shape=(latent_dim,),
                bounds=self.prob_bounds,
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
            "P(S|L)": (
                self.params["P(S|L)"] / torch.sum(self.params["P(S|L)"], dim=0)[None, :]
            ),
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
    """
    This model is parameterically conservative and structurally radical.

    The joint probability of binary measurement outcomes X and Y for settings S and T is,
    :math:`P_{XY|ST} = \\sum_{\\lambda \in \\Lambda} P_{X|S\\lambda} P_{Y|ST\\lambda} P_{S | \\lambda} P_{\\lambda}`

    Number of parameters:
    :math:`|\\Lambda | (m^2 + m + 1)`
    """

    def __init__(self, n_settings: int, latent_dim: int, **kwargs):
        super().__init__(n_settings, latent_dim, **kwargs)

        self.terms = {
            "P(X=0|STL)": dict(
                shape=(n_settings, n_settings, latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(Y=0|TL)": dict(
                shape=(n_settings, latent_dim),
                bounds=self.prob_bounds,
            ),
            "P(L)": dict(
                shape=(latent_dim,),
                bounds=self.prob_bounds,
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
    device = "cpu"

    io = IO.directory(
        folder="entangled-state-data", include_date=False, include_id=False
    )

    run = 0
    m = 80
    p = 0.9
    latent_dim = 30
    n_steps = 300
    lr = 0.25

    data = torch.Tensor(io.load_np_array(filename=f"m={m}_p={p}_{run}.npy")).to(device)
    data_test = torch.Tensor(io.load_np_array(filename=f"m={m}_p={p}_{(run+1)%3}.npy")).to(device)

    training_curves = {}

    for Model in [
        ClassicalCommonCause,
        Superdeterminism,
        Superluminal,
        QuantumCommonCause
    ]:
        model = Model(n_settings=m, latent_dim=latent_dim if Model is not QuantumCommonCause else 2)

        # for _ in range(3):
        pred = model.forward()

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # loss = torch.nn.MSELoss()
        # loss = torch.nn.KLDivLoss()
        loss = KLDivLoss()

        t0 = time.time()
        losses = fit(model, data, optimizer, loss, n_steps=n_steps)
        test = model.forward()

        training_curves[model.__class__.__name__] = losses

        print(
            f"\n{model.__class__.__name__} | "
            f"\n\tTotal time: {time.time() - t0}| "
            f"\n\tTotal parameters: {sum(p.numel() for p in model.parameters())}"
            f"\n\tTrain loss: {loss(model.forward(), data)}"
            f"\n\tTest loss: {loss(model.forward(), data_test)}"
        )

        torch.cuda.empty_cache()

        # fig, axs = plt.subplots(nrows=1, ncols=2)
        # k = 5
        # axs[0].imshow(to_numpy(data[:, :, :k, :k]).reshape([4, k ** 2]))
        # axs[1].imshow(to_numpy(pred[:, :, :k, :k]).reshape([4, k ** 2]))
        #
        # plt.show()
    #%%
    fig, ax = plt.subplots(1, 1)
    for label, losses in training_curves.items():
        ax.plot(np.arange(len(losses)), losses, label=f"{label}")
        ax.set_yscale('log')

    ax.legend()
    plt.show()
