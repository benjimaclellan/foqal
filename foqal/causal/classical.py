import torch
from torch import nn

from foqal.model import ModelBase


class ClassicalProbabilityCausalModel(ModelBase):

    def __init__(self, num_settings: int, latent_dim: int, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_settings = num_settings
        self.latent_dim = latent_dim

    def initialize_params(self):
        self.params = nn.ParameterDict()

        for key, term in self.terms.items():
            self.params[key] = nn.Parameter(
                torch.distributions.Uniform(
                    low=term["bounds"][0], high=term["bounds"][1]
                ).sample(term["shape"])
            )


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