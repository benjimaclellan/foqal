import qutip as qt
import numpy as np
import itertools
import warnings

from foqal.utils.io import IO
from foqal.utils.sample import distribute_points_on_sphere, bloch_vectors_to_kets
from foqal.utils.constants import states, channels


def simulate_quantum_states_effects(
        num_states: int = 10,
        num_effects: int = 10,
        n_parties: int = 1,
        dim: int = 2,
        method: str = "haar",

):

    if dim > 2 and method is not "haar":
        warnings.warn("Must use Haar random sampling for dim>2.")
        method = "haar"

    if method in ("fibonnaci", "spiral"):
        _states = bloch_vectors_to_kets(
            distribute_points_on_sphere(num_samples=num_states, method=method)
        )
        _effects = bloch_vectors_to_kets(
            distribute_points_on_sphere(num_samples=num_effects, method=method)
        )

        states = [qt.tensor(*s) for s in itertools.product(_states, repeat=n_parties)]
        effects = [qt.tensor(*e) for e in itertools.product(_effects, repeat=n_parties)]

    elif method == "haar":
        states = [qt.rand_ket(dim ** n_parties, dims=[n_parties * [dim,], n_parties * [1,]]) for _ in range(num_states)]
        effects = [qt.rand_ket(dim ** n_parties, dims=[n_parties * [dim,], n_parties * [1,]]) for _ in range(num_effects)]

    data = np.zeros([num_states, num_effects])
    for i, state in enumerate(states):
        for j, effect in enumerate(effects):
            data[i, j] = np.abs(np.squeeze((state.dag() * effect).full())) ** 2
    return data


if __name__ == "__main__":

    io = IO.directory(
        folder="gpt-generated-data", verbose=False, include_date=False, include_uuid=False
    )

    n_parties = 1
    # num = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    num = [50, 100]

    for dim in (2, 3, 4):
        for run in (0, 1):

            for i, (num_states, num_effects) in enumerate(zip(num, num)):
                print(f"Run {run} | {i} of {len(num)} | num_states={num_states} | num_effects={num_effects}")

                data = simulate_quantum_states_effects(
                    num_states=num_states, num_effects=num_effects, method="fibonnaci", n_parties=n_parties, dim=dim,
                )

                io.save_np_array(data.astype("float"), filename=f"dim={dim}_n_parties={n_parties}_num_states={num_states}_num_effects={num_effects}_run{run}")
