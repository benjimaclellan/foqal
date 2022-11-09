import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

from foqal.utils.io import IO
from foqal.utils.style import StyleConfig


def model_comparison(df: pd.DataFrame):
    """
    Generates a plot of test & train losses for classical & quantum causal models, as a function of depolarizing
    coefficient, p (x-axis) and number of measurements (line color).

    :param df: pandas.DataFrame containing all the regression results.
    :return:
    """
    config = StyleConfig()  # useful for consistent plot types, colors, styles, etc.
    verbose = False

    names = {  # label to give to each model in the legend
        "ClassicalCommonCauseModel": "CCK",
        "SuperluminalModel": "SL",
        "SuperdeterminismModel": "SD",
        "QuantumCommonCauseModel": "qCC",
    }

    fig, axs = config.grid_axes(
        nrows=1,
        ncols=2,
        width_ax=100,
        height_ax=70,
        left=25,
        right=65,
        bottom=15,
        top=10,
        width_space=15,
        height_space=0,
        sharex=True,
        sharey=False,
        squeeze=True,
    )

    ms = df["m"].unique()
    ps = df["p"].unique()

    def scale(m):
        k = 0.6
        return ((m - np.min(ms)) / (np.max(ms) - np.min(ms))) * k + (1 - k) / 2

    models = [
        "ClassicalCommonCause",
        "Superluminal",
        "Superdeterminism",
        "QuantumCommonCause",
    ]
    colors = ["Reds", "Blues", "Greens", "Greys"]

    lines = [axs[0].plot([], marker="", ls="", label=" ")]
    lines += [axs[0].plot([], marker="", ls="", label=name) for name in names.values()]

    for m in ms:
        lines += [axs[0].plot([], marker="", ls="", label=r"$m$=" + f"{m} \t")]
        for model, c in zip(models, colors):
            train, test = [], []
            for p in ps:
                latent_dim = 50
                dfj = df[(df["model"] == model) & (df["m"] == m) & (df["p"] == p)]

                if verbose:
                    print(model, p)
                    print(latent_dim)
                    print(dfj.shape[0])

                dfi = df[(df["model"] == model) & (df["m"] == m) & (df["p"] == p)]
                dfj = dfi[(dfi["train_loss"] == dfi["train_loss"].min())]
                train.append(dfj[f"train_loss"].mean())
                test.append(dfj[f"test_loss"].mean())

            cmap = plt.get_cmap(c)

            (l,) = axs[0].plot(ps, train, ls="-", color=cmap(scale(m)), label=" ")
            lines.append(l)

            axs[1].plot(ps, test, ls="--", color=cmap(scale(m)))
    axs[0].set(xlabel=r"Depolarizing coefficient, $p$", ylabel="Training error")
    axs[1].set(xlabel=r"Depolarizing coefficient, $p$", ylabel="Test error")

    # messing around with the legend
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    handles, labels = axs[0].get_legend_handles_labels()
    leg = axs[1].legend(
        flip(handles, 5),
        flip(labels, 5),
        ncol=5,
        handletextpad=0.1,
        columnspacing=0.3,
        labelspacing=0.1,
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
    )
    leg._legend_box.align = "center"

    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    return fig, axs


if __name__ == "__main__":
    io = IO.directory(
        folder="ibmq-simulator_bell-state_local-projections_depolarized-channel",
        include_date=False,
        include_id=False,
        verbose=False,
    )

    df = io.load_dataframe("model_summary.txt")
    print(df)
    model_comparison(df)
