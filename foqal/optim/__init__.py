from torch.nn import MSELoss

from torch import Tensor
from torch.nn import KLDivLoss as _KLDivLoss
from torch.nn.functional import kl_div


"""
https://vene.ro/blog/mirror-descent.html
"""


class KLDivLoss(_KLDivLoss):
    r"""
    Implements the KL divergence, without requiring log-probabilities in the input.
    """
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # KL Divergence without requiring log probabilities
        return kl_div(
            input.log(), target, reduction=self.reduction, log_target=self.log_target
        )
