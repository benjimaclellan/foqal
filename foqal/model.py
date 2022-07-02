import torch
from torch import nn


class ModelBase(nn.Module):

    """

    """
    def __init__(self, **kwargs):

        super().__init__()

        self.verbose = (
            kwargs.get("verbose") if kwargs.get("verbose") is not None else False
        )

        self.terms = {}
        self.params = None

        return

    def forward(self):
        """
        This will take the (N,)-shaped parameter vector and construct the predicted
        values. This involves any reshaping, manual constraints, and the general
        mathematic function of the model. Terms are accessed by their dictionary
        key, i.e. p['Px|y']. Checks should be put in place to ensure the shapes are
        correct and that the model is functioning as expected

        From Patrick's thesis:
        We will have a data matrix, D, where each row is the frequency vector F(x, y)
        Each row F is the relative frequency of (x,y)= (0,0) or (0,1) or (1,0), or (1,1)
        So we will need to 'predict' a 4xN matrix
        The number of settings is the number of s and t (keep the same number for both)

        pars: parameter vector used in (to match scipy.optimize.minimize)

        Returns
        pred: the predicted values (in whatever shape), which can be compared to
        data
        """
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