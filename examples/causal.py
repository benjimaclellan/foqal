import time
import itertools
import tqdm
import pandas as pd
import torch
from torch.functional import F
import numpy as np
import matplotlib.pyplot as plt


from foqal.utils.io import IO
from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal
from foqal.causal.quantum import QuantumCommonCause
from foqal.fit import fit


if __name__ == "__main__":
    use_device = True
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    device = torch.cuda.current_device()

    io = IO.directory(
        folder="entangled-state-data", include_date=False, include_uuid=False, verbose=False,
    )
    
    # ms = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,)
    ms = (70,)

    latent_dim = 30
    lr = 0.05
    n_steps = 300

    q = list(itertools.product(ms, ps))

    verbose, show = False, False
    df = []
    for i in (pbar := tqdm.tqdm(range(len(q)))):
        (m, p) = q[i]
        pbar.set_description(f"m={m} | p={p}")

        train_data = torch.Tensor(io.load_np_array(filename=f"m={m}_p={p}_run{0}.npy"))
        test_data = torch.Tensor(io.load_np_array(filename=f"m={m}_p={p}_run{1}.npy"))
        if use_device:
            train_data = train_data.to(device)
            test_data = test_data.to(device)

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

            model = Model(num_settings=m, latent_dim=_latent_dim)

            if use_device:
                model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            t0 = time.time()
            losses = fit(model, train_data, optimizer, n_steps=n_steps, progress=False)
            t1 = time.time()

            if verbose:
                print(
                    f"\n{model.__class__.__name__} | "
                    f"\n\tTotal time: {t1 - t0}| "
                    f"\n\tTotal parameters: {sum(p.numel() for p in model.parameters())}"
                    f"\n\tFinal loss: {losses[-1]}"
                )

            torch.cuda.empty_cache()

            loss_test = F.mse_loss(model.forward(), test_data)
            if loss_test.is_cuda:
                loss_test = loss_test.cpu().detach().numpy().item()
            else:
                loss_test = loss_test.detach().numpy().item()

            df.append(dict(
                model=model.__class__.__name__,
                m=m,
                p=p,
                latent_dim=_latent_dim,
                train_loss=losses[-1].item(),
                test_loss=loss_test,
                # train_curve=losses,
                t=(t1 - t0),
                lr=lr,
                n_steps=n_steps,
            ))

    io.verbose = True
    df = pd.DataFrame(df)
    io.save_dataframe(df, filename="summary_of_fitting.txt")