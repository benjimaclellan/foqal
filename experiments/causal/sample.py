import qutip as qt
import numpy as np

from foqal.utils.io import IO
from foqal.utils.sample import sample_bloch_vectors, bloch_vectors_to_kets
from foqal.utils.constants import channels


def simulate_local_projective_measurements(
    state: qt.Qobj,
    n_datasets: int = 2,
    m: int = 10,
    method: str = "fibonnaci",
    total_counts=2000,
):
    """
    Simulates two-qubit measurement outcome probabilities for pairs of local, sampled projective measurements.

    n datasets are sampled, using a Poissonian noise model.

    :param state: quantum state, as a qt.Qobj, to simulate the measurement outcomes of
    :param n_datasets: number of sampled datasets to return (useful for cross-validation techniques)
    :param m: number of unique projective measurements applied locally on each qubit
    :param method: sampling method. options are "haar", "fibonacci"
    :param total_counts: total number of measurement counts (i.e., number of photon events)
    :return:
    """
    assert method in ("fibonnaci", "haar")

    settings = bloch_vectors_to_kets(sample_bloch_vectors(num_samples=m, method=method))
    settings = [qt.ket2dm(setting) for setting in settings]

    data = np.zeros([2, 2, m, m])
    eye = qt.qeye(2)
    for s, setting_s in enumerate(settings):
        for t, setting_t in enumerate(settings):
            data[0, 0, s, t] = (qt.tensor(setting_s, setting_t) * state).tr()
            data[0, 1, s, t] = (qt.tensor(setting_s, eye - setting_t) * state).tr()
            data[1, 0, s, t] = (qt.tensor(eye - setting_s, setting_t) * state).tr()
            data[1, 1, s, t] = (
                qt.tensor(eye - setting_s, eye - setting_t) * state
            ).tr()

    datasets = []
    for j in range(n_datasets):
        _data = np.random.poisson(total_counts * data)
        _data = _data / np.sum(_data, axis=(0, 1))[None, None, :, :]
        datasets.append(_data)
    return datasets


if __name__ == "__main__":

    io = IO.directory(
        folder="simulated-data-causal-two-qubit-depolarizing",
        verbose=True,
        include_date=False,
        include_id=False,
    )
    n_datasets = 2
    ms = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140)
    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    for m in ms:
        for p in ps:
            state = channels["depolarized"](p=p)
            datasets = simulate_local_projective_measurements(
                state=state,
                n_datasets=n_datasets,
                m=m,
                method="fibonnaci",
            )

            for k, data in enumerate(datasets):
                assert np.all(np.isclose(np.sum(data, axis=(0, 1)), 1.0))

                io.save_np_array(
                    data.astype("float"),
                    filename=f"num_states={m}_p={int(100 * p)}_{k}",
                )
