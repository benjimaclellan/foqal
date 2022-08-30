import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd

from foqal.utils.io import IO

from foqal.utils.style import StyleConfig


def model_comparison(df: pd.DataFrame):

    config = StyleConfig()  # useful for consistent plot types, colors, styles, etc.
    verbose = False

    # io = IO(path=paths.path_main.joinpath(paths.paths_simulations['num_measurements']))  # this is the good dataset
    # df = io.load_dataframe("model-performance.txt")

    names = {  # label to give to each model in the legend
        "ClassicalCommonCauseModel": 'CCK',
        "SuperluminalModel": "SL",
        "SuperdeterminismModel": "SD",
         "QuantumCommonCauseModel": 'qCC',
    }

    fig, axs = config.grid_axes(nrows=2, ncols=1, width_ax=90, height_ax=30, left=25, right=55, bottom=15, top=5,
                                width_space=10, height_space=5, sharex=True, sharey=False, squeeze=True)

    ms = df['m'].unique()
    ps = df['p'].unique()

    def scale(m):
        k = 0.8
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
        lines += [axs[0].plot([], marker="", ls="", label=r"$m$="+f"{m} \t")]
        for model, c in zip(models, colors):
            train, test = [], []
            for p in ps:
                latent_dim = 50
                dfj = df[(df['model'] == model)
                         & (df['m'] == m)
                         & (df['p'] == p)
                         ]
                # row = dfj[dfj['train_loss'] == dfj['train_loss'].min()]

                if verbose:
                    print(model, p)
                    print(latent_dim)
                    print(dfj.shape[0])

                dfi = df[(df['model'] == model) & (df['m'] == m) & (df['p'] == p)]
                train.append(dfi[f'train_loss'].min())
                test.append(dfi[f'test_loss'].min())

            cmap = plt.get_cmap(c)
            # norm = m * m * 4
            norm = 1

            l, = axs[0].plot(ps, train, ls='-', color=cmap(scale(m)), label=" ")  # label=f"{m}")
            lines.append(l)

            axs[1].plot(ps, test, ls='--', color=cmap(scale(m)))  # , label=f"Test: {model} MEAN")

    axs[0].set(ylabel="Training error")
    axs[1].set(xlabel=r"Depolarizing coefficient, $p$", ylabel="Test error")

    for ax in axs:
        ax.set(ylim=[0, 0.0003])

    # messing around with the legend
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    handles, labels = axs[0].get_legend_handles_labels()
    leg = axs[0].legend(flip(handles, 5), flip(labels, 5),
                        ncol=5,
                        handletextpad=0.1,
                        columnspacing=0.3,
                        labelspacing=0.1,
                        bbox_to_anchor=(0.95, 0), loc="center left")
    leg._legend_box.align = "center"

    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    # save and show the figure
    # config.save_figure(fig, "fig_causal_discovery_simulations_n_measurements")
    plt.show()


if __name__ == "__main__":
    io = IO.directory(
        folder="ibmq-simulator_bell-state_local-projections_depolarized-channel",
        # folder="2022-08-23_cross_val_5ccc",
        include_date=False, include_id=False, verbose=False,
    )

    df = io.load_dataframe("model_summary.txt")
    print(df)
    model_comparison(df)