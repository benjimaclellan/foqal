from torch.nn import MSELoss
from torch import Tensor
from torch.nn import KLDivLoss as _KLDivLoss
import torch.nn.functional as F


class KLDivLoss(_KLDivLoss):
    r"""The Kullback-Leibler divergence loss.

    For tensors of the same shape :math:`y_{\text{pred}},\ y_{\text{true}}`,
    where :math:`y_{\text{pred}}` is the :attr:`input` and :math:`y_{\text{true}}` is the
    :attr:`target`, we define the **pointwise KL-divergence** as

    .. math::

        L(y_{\text{pred}},\ y_{\text{true}})
            = y_{\text{true}} \cdot \log \frac{y_{\text{true}}}{y_{\text{pred}}}
            = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})

    To avoid underflow issues when computing this quantity, this loss expects the argument
    :attr:`input` in the log-space. The argument :attr:`target` may also be provided in the
    log-space if :attr:`log_target`\ `= True`.

    To summarise, this function is roughly equivalent to computing

    .. code-block:: python

        if not log_target: # default
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

    and then reducing this result depending on the argument :attr:`reduction` as

    .. code-block:: python

        if reduction == "mean":  # default
            loss = loss_pointwise.mean()
        elif reduction == "batchmean":  # mathematically correct
            loss = loss_pointwise.sum() / input.size(0)
        elif reduction == "sum":
            loss = loss_pointwise.sum()
        else:  # reduction == "none"
            loss = loss_pointwise

    .. note::
        As all the other losses in PyTorch, this function expects the first argument,
        :attr:`input`, to be the output of the model (e.g. the neural network)
        and the second, :attr:`target`, to be the observations in the dataset.
        This differs from the standard mathematical notation :math:`KL(P\ ||\ Q)` where
        :math:`P` denotes the distribution of the observations and :math:`Q` denotes the model.

    .. warning::
        :attr:`reduction`\ `= "mean"` doesn't return the true KL divergence value, please use
        :attr:`reduction`\ `= "batchmean"` which aligns with the mathematical definition.
        In a future release, `"mean"` will be changed to be the same as `"batchmean"`.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to `False`, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is `False`. Default: `True`
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is `False`, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: `True`
        reduction (string, optional): Specifies the reduction to apply to the output. Default: `"mean"`
        log_target (bool, optional): Specifies whether `target` is the log space. Default: `False`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar by default. If :attr:`reduction` is `'none'`, then :math:`(*)`,
          same shape as the input.

    Examples::

        >>> kl_loss = nn.KLDivLoss(reduction="batchmean")
        >>> # input should be a distribution in the log space
        >>> input = F.log_softmax(torch.randn(3, 5, requires_grad=True))
        >>> # Sample a batch of distributions. Usually this would come from the dataset
        >>> target = F.softmax(torch.rand(3, 5))
        >>> output = kl_loss(input, target)

        >>> kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        >>> log_target = F.log_softmax(torch.rand(3, 5))
        >>> output = kl_loss(input, log_target)
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # KL Divergence without requiring log probabilities
        return F.kl_div(input.log(), target, reduction=self.reduction, log_target=self.log_target)
