import time
import itertools
import tqdm
import pandas as pd
import torch

from foqal import optim
from foqal.utils.io import IO
from foqal.utils import to_numpy
from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal
from foqal.causal.quantum import QuantumCommonCause
from foqal.fit import fit


"""
Main regression script for fitting causal models to measurement data of depolarized Bell states.
"""

if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    device = "cuda"

    io = IO.directory(
        folder="simulated-data-causal-two-qubit-depolarizing",
        include_date=False,
        include_id=False,
        verbose=False,
    )

    ms = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
    # ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ps = (0.9,)

    q = list(itertools.product(ms, ps))

    verbose, show = False, False
    df = []
    curves = []

    for i in (pbar := tqdm.tqdm(range(len(q)))):
        (m, p) = q[i]
        pbar.set_description(f"m={m} | p={p}")

        lr = 0.10
        n_steps = 2000 + m * 40  # change number of steps depending on m

        train_data = torch.Tensor(io.load_np_array(filename=f"m={m}_p={int(100 * p)}_{0}.npy")).to(
            device
        )
        test_data = torch.Tensor(
            io.load_np_array(filename=f"m={m}_p={int(100 * p)}_{1}.npy")
        ).to(device)

        for Model in [
            # ClassicalCommonCause,
            # Superdeterminism,
            # Superluminal,
            QuantumCommonCause,
        ]:
            if Model is QuantumCommonCause:
                _latent_dim = 2
            else:
                _latent_dim = max([15, m])

            check = True
            count = 0
            while check:

                model = Model(n_settings=m, latent_dim=_latent_dim)
                model = model.to(device)

                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
                loss = optim.KLDivLoss()

                t0 = time.time()
                losses = fit(
                    model, train_data, optimizer, loss, n_steps=n_steps, progress=False
                )
                t1 = time.time()

                loss_train = to_numpy(loss(model.forward(), train_data))
                loss_test = to_numpy(loss(model.forward(), test_data))

                if (Model == QuantumCommonCause) and (loss_train > 0.0002):
                    print("Not good enough - running again.")
                    check = True
                    count += 1
                    if count > 10:
                        print(f"Not converging well")
                        check = False
                else:
                    check = False

            df.append(
                dict(
                    model=model.__class__.__name__,
                    m=m,
                    p=p,
                    latent_dim=_latent_dim,
                    train_loss=loss_train,
                    test_loss=loss_test,
                    kl_test_train=to_numpy(loss(train_data, test_data)),
                    t=(t1 - t0),
                    lr=lr,
                    n_steps=n_steps,
                )
            )
            # curves.append(losses)

        # save intermediate results
        # if i % len(ps):
        #     io.verbose = True
        #     io.save_dataframe(pd.DataFrame(df), filename="results/regression_summary-quantum.txt")
        #     # io.save_json(curves, filename="results/training_curves")
        #     io.verbose = False

    df = pd.DataFrame(df)
    io.verbose = True
    io.save_dataframe(df, filename="results/regression_summary-quantum.txt")
    # io.save_json(curves, filename="results/training_curves")
    print("Regression finished.")
