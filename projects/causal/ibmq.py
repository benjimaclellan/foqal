"""
Implementation of depolarizing channel on ibmQ
https://matteoacrossi.github.io/oqs-jupyterbook/project_1-solution.html
"""
import itertools
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute, ClassicalRegister
import matplotlib.pyplot as plt

from foqal.utils.io import IO


def depolarizing_channel(q, c, p, system=1, ancillae=(2, 3, 4)):
    """Returns a QuantumCircuit implementing depolarizing channel on q[system]

    Args:
        q (QuantumRegister): the register to use for the circuit
        p (float): the probability for the channel between 0 and 1
        system (int): index of the system qubit
        ancillae (list): list of indices for the ancillary qubits

    Returns:
        A QuantumCircuit object
    """

    dc = QuantumCircuit(q, c)

    theta = 1 / 2 * np.arccos(1 - 2 * p)

    dc.ry(theta, q[ancillae[0]])
    dc.ry(theta, q[ancillae[1]])
    dc.ry(theta, q[ancillae[2]])

    dc.cx(q[ancillae[0]], q[system])
    dc.cy(q[ancillae[1]], q[system])
    dc.cz(q[ancillae[2]], q[system])

    return dc


def bell_state(q, c):
    bs = QuantumCircuit(q, c)
    bs.h(q[0])  # Hadamard gate
    bs.cx(q[0], q[1])  # CNOT gate
    return bs


def rotate_measurement_basis(q, c, basis0, basis1):
    rm = QuantumCircuit(q, c)

    for i, rots in {0: basis0, 1: basis1}.items():
        rm.rx(rots[0], q[i])
        rm.ry(rots[1], q[i])
        rm.rz(rots[2], q[i])
    return rm


def sample_rotations():
    return np.random.uniform(0, 2 * np.pi, 3)


def trace_out_ancillae(counts: dict):
    partial = {"00": 0, "01": 0, "10": 0, "11": 0}
    for basis, count in counts.items():
        partial[basis[-2:]] += count
    return partial


# Prepare the qubit in a state that has coherence and different populations
q = QuantumRegister(5, "q")
c = ClassicalRegister(5, "c")

# p_values = np.linspace(0, 1, 10)
# ms = (5,)
# ps = (0.0,)
# n_datasets = 1

n_datasets = 5
ms = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

io = IO.directory(
        folder="ibmq-simulator_bell-state_local-projections_depolarized-channel",
        verbose=True,
        include_date=False,
        include_id=False,
    )

# Here we will create a list of results for each different value of p
circuits = []

for m in ms:
    rotations = []
    for i in range(m):
        rotations.append(sample_rotations())

    for p in ps:
        for k in range(n_datasets):
            data = np.zeros([2, 2, m, m])
            for i, j in itertools.product(range(m), range(m)):
                circ = (
                    bell_state(q, c)
                    + depolarizing_channel(q, c, p)
                    + rotate_measurement_basis(q, c, rotations[i], rotations[j])
                )
                circ.measure_all(add_bits=False)

                job = execute(circ, Aer.get_backend("qasm_simulator"), shots=10000)
                counts = trace_out_ancillae(job.result().get_counts())
                print(
                    f"m={m}, p={p}, k={k}| i={i}, j={j} | {counts}"
                )

                data[0, 0, i, j] = counts["00"]
                data[0, 1, i, j] = counts["10"]
                data[1, 0, i, j] = counts["01"]
                data[1, 1, i, j] = counts["11"]

            data = data / np.sum(data, axis=(0, 1))

            io.save_np_array(data, filename=f"m={m}_p={p}_{k}")
