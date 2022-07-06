import qutip as qt
import numpy as np

from foqal.io import IO
from foqal.utils.sample import distribute_points_on_sphere, bloch_vectors_to_kets
from foqal.utils.constants import channels


def simulate_joint_measurements(
    state: qt.Qobj,
    n_datasets: int = 2,
    n_settings: int = 10,
    method: str = "fibonnaci",
):
    assert method in ("fibonnaci", "haar")

    settings = bloch_vectors_to_kets(
        distribute_points_on_sphere(num_samples=n_settings, method=method)
    )
    settings = [qt.ket2dm(setting) for setting in settings]

    data = np.zeros([2, 2, n_settings, n_settings])
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
    total_counts = 300
    for j in range(n_datasets):
        _data = np.random.poisson(total_counts * data)
        _data = _data / np.sum(_data, axis=(0, 1))[None, None, :, :]
        datasets.append(_data)
    return datasets


if __name__ == "__main__":

    io = IO.directory(
        folder="causal-generated-data",
        verbose=True,
        include_date=False,
        include_uuid=False,
    )
    n_datatsets = 5
    n_settings = 30

    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    for p in ps:
        state = channels["depolarized"](p=p)
        datasets = simulate_joint_measurements(
            state=state,
            n_datasets=n_datatsets,
            n_settings=n_settings,
            method="fibonnaci",
        )

        for k, data in enumerate(datasets):
            assert np.all(np.isclose(np.sum(data, axis=(0, 1)), 1.0))

            io.save_np_array(
                data.astype("float"),
                filename=f"num_states={n_settings}_p={int(100 * p)}_{k}",
            )
