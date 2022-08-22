import qutip as qt
import numpy as np
import itertools
import warnings

from foqal.utils.io import IO
from foqal.utils.sample import sample_bloch_vectors, bloch_vectors_to_kets


def simulate_quantum_states_effects(
    n_datasets=2,
    n_states: int = 10,
    n_effects: int = 10,
    n_parties: int = 1,
    dim: int = 2,
    method: str = "haar",
):

    if dim > 2 and method is not "haar":
        warnings.warn("Must use Haar random sampling for dim>2.")
        method = "haar"

    if method in ("fibonnaci", "spiral"):
        _states = bloch_vectors_to_kets(
            sample_bloch_vectors(num_samples=n_states, method=method)
        )
        _effects = bloch_vectors_to_kets(
            sample_bloch_vectors(num_samples=n_effects, method=method)
        )

        states = [qt.tensor(*s) for s in itertools.product(_states, repeat=n_parties)]
        effects = [qt.tensor(*e) for e in itertools.product(_effects, repeat=n_parties)]

    # autoformat off
    elif method == "haar":
        _dims = [
            n_parties * [dim],
            n_parties * [1],
        ]
        states = [qt.rand_ket(dim**n_parties, dims=_dims) for _ in range(n_states)]
        effects = [qt.rand_ket(dim**n_parties, dims=_dims) for _ in range(n_effects)]

    data = np.zeros([n_states, n_effects])
    for i, state in enumerate(states):
        for j, effect in enumerate(effects):
            data[i, j] = np.abs(np.squeeze((state.dag() * effect).full())) ** 2

    datasets = []
    total_counts = 300
    for j in range(n_datasets):
        # _data = np.random.poisson(total_counts * data) / total_counts
        _data = np.random.normal(data, scale=0.1)
        datasets.append(_data)
    return datasets


if __name__ == "__main__":

    io = IO.directory(
        folder="gpt-generated-data",
        verbose=False,
        include_date=False,
        include_id=False,
    )

    n_parties = 1
    # num = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    num = [50, 100]

    for dim in (
        # 2,
        # 3,
        # 4,
        5,
        6,
    ):

        for i, (num_states, num_effects) in enumerate(zip(num, num)):
            print(
                f"{i} of {len(num)} | num_states={num_states} | num_effects={num_effects}"
            )

            datasets = simulate_quantum_states_effects(
                n_datasets=2,
                n_states=num_states,
                n_effects=num_effects,
                method="haar",
                n_parties=n_parties,
                dim=dim,
            )

            for run, data in enumerate(datasets):
                io.save_np_array(
                    data.astype("float"),
                    filename=f"dim={dim}_n_parties={n_parties}_num_states={num_states}_num_effects={num_effects}_run{run}",
                )
