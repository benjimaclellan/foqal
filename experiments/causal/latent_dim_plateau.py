import time
import itertools
import tqdm
import pandas as pd
import torch
import ray

from foqal import optim
from foqal.utils.io import IO
from foqal.utils import to_numpy
from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal
from foqal.fit import fit


"""
Sweep through |Lambda| for classical models
"""

if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    device = "cpu"

    io = IO.directory(
        folder="simulated-data-causal-two-qubit-depolarizing",
        include_date=False,
        include_id=False,
        verbose=False,
    )

    remote = True
    n_cpus = 17

    lr = 0.10
    n_steps = 2000  # change number of steps depending on m
    m = 50
    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ks = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)

    models = (ClassicalCommonCause, Superdeterminism, Superluminal)

    q = list(itertools.product(ps, ks, models))

    verbose, show = False, False
    df = []

    train_datas = {
        p: torch.Tensor(
            io.load_np_array(filename=f"m={m}_p={int(100 * p)}_{0}.npy")
        ).to(device)
        for p in ps
    }
    test_datas = {
        p: torch.Tensor(
            io.load_np_array(filename=f"m={m}_p={int(100 * p)}_{1}.npy")
        ).to(device)
        for p in ps
    }

    def run(model, p):
        print(f"m={model.n_settings} | p={p} | k={model.latent_dim}")
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        loss = optim.KLDivLoss()
        t0 = time.time()

        train_data = train_datas[p]
        test_data = test_datas[p]

        _ = fit(model, train_data, optimizer, loss, n_steps=n_steps, progress=False)
        t1 = time.time()

        loss_train = to_numpy(loss(model.forward(), train_data))
        loss_test = to_numpy(loss(model.forward(), test_data))

        d = dict(
            model=model.__class__.__name__,
            m=m,
            p=p,
            latent_dim=model.latent_dim,
            train_loss=loss_train,
            test_loss=loss_test,
            kl_test_train=to_numpy(loss(train_data, test_data)),
            t=(t1 - t0),
            lr=lr,
            n_steps=n_steps,
        )
        return d

    if remote:
        ray.init(num_cpus=n_cpus, ignore_reinit_error=True)
        run = ray.remote(run)

    futures = []
    for i in (pbar := tqdm.tqdm(range(len(q)))):
        (p, k, Model) = q[i]
        # pbar.set_description(f"m={m} | p={p} | k={k}")
        model = Model(n_settings=m, latent_dim=k)
        model = model.to(device)

        futures.append(run.remote(model, p) if remote else run(model, p))

    if remote:
        futures = ray.get(futures)

    df = pd.DataFrame(futures)

    io.verbose = True
    io.save_dataframe(df, filename="heatmap/latent_dim_plateau.txt")

    print("Regression finished.")
