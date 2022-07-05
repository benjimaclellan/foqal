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
from foqal.fit import fit, kfold_validation


if __name__ == "__main__":
    use_device = True
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    device = torch.cuda.current_device()

    io = IO.directory(
        folder="causal-generated-data", include_date=False, include_uuid=False, verbose=False,
    )

    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,)
    m = 30

    latent_dim = 100
    lr = 0.25
    n_steps = 300
    n_datasets = 5

    verbose, show = False, False
    df = []
    for i, p in enumerate(ps):

        datasets = [
            torch.Tensor(io.load_np_array(filename=f"num_states={m}_p={int(100*p)}_{k}.npy")).to(device) for k in range(n_datasets)
        ]

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
            print(f"\n\nBeginning k-fold validation of p={p} | {Model.__name__}")
            model = Model(n_settings=m, latent_dim=_latent_dim).to(device)
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

            results = kfold_validation(model, datasets=datasets, optimizer=optimizer, n_steps=n_steps)

            df.append(dict(
                model=model.__class__.__name__,
                m=m,
                p=p,
                latent_dim=_latent_dim,
                train_loss=results['train'].mean(),
                test_loss=results['test_mean'].mean(),
                test_std=results['test_std'].mean(),
                lr=lr,
                n_steps=n_steps,
            ))

    io.verbose = True
    df = pd.DataFrame(df)

    io = IO.directory(
        folder=f"causal-generated-data/{m}", include_date=False, include_uuid=False, verbose=True,
    )
    io.save_dataframe(df, filename="summary_of_fitting.txt")
