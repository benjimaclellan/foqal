import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from foqal.fit import fit, loss_function

from foqal.model import ModelBase
from foqal.utils.io import IO


class GeneralizedProbabilityTheory(ModelBase):
    """
    A generalized probabilistic theory (GPT) is a general framework to describe the operational features of
    arbitrary physical theories.

    :math:`\\mathcal{D} = \\mathbf{S} \\times \\mathbf{E}`

    """

    def __init__(self, n_states: int, n_effects: int, rank: int, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_effects = n_effects
        self.rank = rank

        self.terms = {
            "S": dict(
                shape=(self.n_states, self.rank),
                bounds=(-1, 1),
            ),
            "E": dict(
                shape=(self.rank, self.n_effects),
                bounds=(-1, 1),
            ),
        }

        self.initialize_params()
        return

    def forward(self):
        self.clip_params()
        pred = torch.einsum("ij,jk->ik", self.params["S"], self.params["E"])
        return pred


if __name__ == "__main__":
    use_device = True

    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    device = torch.cuda.current_device()

    io = IO.directory(
        folder="gpt-generated-data", include_date=False, include_uuid=False
    )
    n_states, n_effects = 100, 100
    n_parties = 1
    dim = 5
    ranks = range(1, 50, 1)

    train_data = torch.Tensor(
        io.load_np_array(
            filename=f"dim={dim}_n_parties={n_parties}_num_states={n_states}_num_effects={n_effects}_run{0}.npy"
        )
    )
    test_data = torch.Tensor(
        io.load_np_array(
            filename=f"dim={dim}_n_parties={n_parties}_num_states={n_states}_num_effects={n_effects}_run{1}.npy"
        )
    )
    if use_device:
        train_data = train_data.to(device)
        test_data = test_data.to(device)

    training_curves = {}

    training_loss = {}
    testing_loss = {}

    for rank in ranks:
        model = GeneralizedProbabilityTheory(
            n_states=n_states, n_effects=n_effects, rank=rank
        )

        if use_device:
            model = model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=30)

        t0 = time.time()
        losses = fit(model, train_data, optimizer, n_steps=3000)

        training_curves[rank] = losses
        training_loss[rank] = losses[-1]

        loss = loss_function(model.forward(), test_data)
        if loss.is_cuda:
            loss = loss.cpu().detach().numpy()
        else:
            loss = loss.detach().numpy()
        testing_loss[rank] = loss

        # print(
        #     f"\n{model.__class__.__name__} | "
        #     f"\n\tTotal time: {time.time() - t0}| "
        #     f"\n\tTotal parameters: {sum(p.numel() for p in model.parameters())}"
        #     f"\n\tFinal loss: {losses[-1]}"
        # )

        torch.cuda.empty_cache()

    #%%
    fig, ax = plt.subplots(1, 1)
    for label, losses in training_curves.items():
        ax.plot(np.arange(len(losses)), losses, label=f"k={label}")

    ax.set(xlabel="Training step", ylabel="Loss")
    ax.set(yscale="log")
    # ax.set(ylim=[-0.05, 0.2])
    ax.legend()
    plt.show()

    #%%
    fig, ax = plt.subplots(1, 1)
    ks, ls = zip(*[(k, losses[-1]) for k, losses in training_curves.items()])
    ax.plot(training_loss.keys(), training_loss.values(), label="Train")
    ax.plot(testing_loss.keys(), testing_loss.values(), label="Test")
    ax.set(xlabel="Rank, k", ylabel="Loss")
    ax.legend()
    plt.show()
