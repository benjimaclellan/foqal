import time
import itertools
import tqdm
import pandas as pd
import torch
from torch.functional import F

from foqal.utils.io import IO
from foqal.utils import to_numpy
from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal
from foqal.causal.quantum import QuantumCommonCause
from foqal.fit import fit


if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    device = "cuda"

    io = IO.directory(
        # folder="entangled-state-data",
        folder="ibmq-simulator_bell-state_local-projections_depolarized-channel",
        include_date=False,
        include_id=False,
        verbose=False,
    )

    # ms = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
    ps = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    ms = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80)

    latent_dim = 100
    lr = 0.25
    n_steps = 1000

    q = list(itertools.product(ms, ps))

    verbose, show = False, False
    df = []
    for i in (pbar := tqdm.tqdm(range(len(q)))):
        (m, p) = q[i]
        pbar.set_description(f"m={m} | p={p}")

        train_data = torch.Tensor(
            io.load_np_array(filename=f"m={m}_p={p}_{0}.npy")
        ).to(device)
        test_data = torch.Tensor(
            io.load_np_array(filename=f"m={m}_p={p}_{1}.npy")
        ).to(device)

        for Model in [
            ClassicalCommonCause,
            Superdeterminism,
            Superluminal,
            QuantumCommonCause,
        ]:
            if Model is QuantumCommonCause:
                _latent_dim = 2
            else:
                _latent_dim = latent_dim

            model = Model(n_settings=m, latent_dim=_latent_dim)
            model = model.to(device)

            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
            loss = torch.nn.MSELoss()

            t0 = time.time()
            losses = fit(model, train_data, optimizer, loss, n_steps=n_steps, progress=False)
            t1 = time.time()

            # torch.cuda.empty_cache()
            loss_train = to_numpy(loss(model.forward(), train_data))
            loss_test = to_numpy(loss(model.forward(), test_data))

            df.append(
                dict(
                    model=model.__class__.__name__,
                    m=m,
                    p=p,
                    latent_dim=_latent_dim,
                    train_loss=loss_train,
                    test_loss=loss_test,
                    t=(t1 - t0),
                    lr=lr,
                    n_steps=n_steps,
                )
            )

    df = pd.DataFrame(df)
    io.verbose = True
    io.save_dataframe(df, filename="model_summary.txt")
