import torch
from torch import nn


class ModelBase(nn.Module):

    """ """

    def __init__(self, **kwargs):

        super().__init__()

        self.verbose = (
            kwargs.get("verbose") if kwargs.get("verbose") is not None else False
        )

        self.terms = {}
        self.params = None

        return

    def forward(self):
        """ """
        raise NotImplementedError(
            "Please implement the prediction function for this class"
        )
        return pred

    def clip_params(self):
        with torch.no_grad():
            for key, term in self.terms.items():
                self.params[key][:] = torch.clamp(
                    self.params[key], min=term["bounds"][0], max=term["bounds"][1]
                )

    def initialize_params(self):
        self.params = nn.ParameterDict()

        for key, term in self.terms.items():
            self.params[key] = nn.Parameter(
                torch.distributions.Uniform(
                    low=term["bounds"][0], high=term["bounds"][1]
                ).sample(term["shape"])
            )
