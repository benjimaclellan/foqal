import pandas as pd
import torch
import uuid

from foqal.utils.io import IO
from foqal.causal.classical import ClassicalCommonCause, Superdeterminism, Superluminal
from foqal.causal.quantum import QuantumCommonCause
from foqal.fit import cross_validation


if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    device = "cuda"

    input = IO.directory(
        folder="ibmq-simulator_bell-state_local-projections_depolarized-channel",
        include_date=False,
        include_id=False,
        verbose=False,
    )
    output = IO.directory(
        folder="ibmq-simulator_bell-state_local-projections_depolarized-channel/cross_val",
        include_date=True,
        include_id=True,
        verbose=False,
    )
    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,)
    ms = (5, 10)

    latent_dim = 10
    lr = 0.25
    n_steps = 500
    n_datasets = 5

    verbose, show = False, False
    df = []
    for j, m in enumerate(ms):
        for i, p in enumerate(ps):
            filename = lambda k: f"m={m}_p={p}_{k}.npy"
            datasets = [
                torch.Tensor(input.load_np_array(filename=filename(k))).to(device)
                for k in range(n_datasets)
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
                print(
                    f"\n\nBeginning k-fold validation of m={m}, p={p}, {Model.__name__}"
                )
                model = Model(n_settings=m, latent_dim=_latent_dim).to(device)

                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
                loss = torch.nn.MSELoss()

                uid = uuid.uuid4().hex
                results = cross_validation(
                    model,
                    datasets=datasets,
                    optimizer=optimizer,
                    loss=loss,
                    n_steps=n_steps,
                )

                output.save_np_array(
                    results[0]["training_curve"], filename=f"training_curves/{uid}.npy"
                )

                results = pd.DataFrame(results)
                df.append(
                    dict(
                        model=model.__class__.__name__,
                        m=m,
                        p=p,
                        latent_dim=_latent_dim,
                        train_loss=results["train"].mean(),
                        test_loss=results["test_mean"].mean(),
                        test_std=results["test_std"].mean(),
                        lr=lr,
                        n_steps=n_steps,
                        uid=uid,
                    )
                )

    df = pd.DataFrame(df)
    output.save_dataframe(df, filename="model_summary.txt")
