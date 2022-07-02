import time
import itertools
import tqdm
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


from foqal.utils.io import IO
from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal
from foqal.fit import fit


if __name__ == "__main__":
    use_device = True
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    device = torch.cuda.current_device()

    io = IO.directory(
        folder="entangled-state-data", include_date=False, include_uuid=False
    )
    ms = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    runs = (0, 1, 2)

    latent_dim = 100
    lr = 0.05
    n_steps = 400

    q = list(itertools.product(ms, ps, runs))

    verbose, show = False, False
    df = []
    for i in (pbar := tqdm.tqdm(range(len(q)))):
        (m, p, run) = q[i]

        # load data
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

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            t0 = time.time()
            losses = fit(model, data, optimizer, n_steps=n_steps, progress=False)
            t1 = time.time()

            training_curves[model.__class__.__name__] = losses

            if verbose:
                print(
                    f"\n{model.__class__.__name__} | "
                    f"\n\tTotal time: {t1 - t0}| "
                    f"\n\tTotal parameters: {sum(p.numel() for p in model.parameters())}"
                    f"\n\tFinal loss: {losses[-1]}"
                )

            torch.cuda.empty_cache()

        df.append(dict(
            model=model.__class__.__name__,
            m=m,
            p=p,
            run=run,
            latent_dim=latent_dim,
            t=(t1 - t0),
            lr=lr,
            n_steps=n_steps,
        ))

        if show:  # plot fitting curves
            fig, ax = plt.subplots(1, 1)
            for label, losses in training_curves.items():
                ax.plot(np.arange(len(losses)), np.log(losses), label=f"{label}")

            ax.legend()
            plt.show()

    df = pd.DataFrame(df)
    io.save_dataframe(df, filename="summary_of_fitting.txt")
