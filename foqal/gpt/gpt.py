import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from foqal.fit import fit

from foqal.model import ModelBase
from foqal.utils.io import IO


class GeneralizedProbabilityTheory(ModelBase):
    def __init__(self, num_states: int, num_effects: int, rank: int, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_states = num_states
        self.num_effects = num_effects
        self.rank = rank

        self.terms = {
            "S": dict(
                shape=(self.num_states, self.rank),
                bounds=(-1, 1),
            ),
            "E": dict(
                shape=(self.rank, self.num_effects),
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
    num_states, num_effects = 50, 50
    run = 0
    n_parties = 1
    dim = 4

    data = torch.Tensor(io.load_np_array(filename=f"dim={dim}_n_parties={n_parties}_num_states={num_states}_num_effects={num_effects}_run{run}.npy"))
    if use_device:
        data = data.to(device)

    training_curves = {}

    for rank in range(1, 25):
        model = GeneralizedProbabilityTheory(num_states=num_states, num_effects=num_effects, rank=rank)

        if use_device:
            model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        t0 = time.time()
        losses = fit(model, data, optimizer, n_steps=800)

        training_curves[rank] = losses

        print(
            f"\n{model.__class__.__name__} | "
            f"\n\tTotal time: {time.time() - t0}| "
            f"\n\tTotal parameters: {sum(p.numel() for p in model.parameters())}"
            f"\n\tFinal loss: {losses[-1]}"
        )

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
    ax.plot(ks, ls)
    ax.set(xlabel='Rank, k', ylabel='Loss')

    plt.show()
