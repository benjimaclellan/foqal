"""
Implementation of depolarizing channel on ibmQ
https://matteoacrossi.github.io/oqs-jupyterbook/project_1-solution.html
"""
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute, ClassicalRegister
import matplotlib.pyplot as plt


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


def rotate_measurement_basis(q, c):
    rm = QuantumCircuit(q, c)

    for i in (0, 1):
        rm.rx(np.random.uniform(0, 2*np.pi), q[i])
        rm.ry(np.random.uniform(0, 2*np.pi), q[i])
        rm.rz(np.random.uniform(0, 2*np.pi), q[i])
    return rm


def trace_out_ancillae(counts: dict):
    partial = {"00": 0, "01": 0, "10": 0, "11": 0}
    for basis, count in counts.items():
        partial[basis[-2:]] += count
    return partial


# Prepare the qubit in a state that has coherence and different populations
q = QuantumRegister(5, 'q')
c = ClassicalRegister(5, 'c')

# p_values = np.linspace(0, 1, 10)
ps = (1.0, )

# Here we will create a list of results for each different value of p
circuits = []

for p in ps:
    circ = bell_state(q, c) + depolarizing_channel(q, c, p) + rotate_measurement_basis(q, c)
    circ.measure_all(add_bits=False)
    circuits.append(circ)

results = []
for circ in circuits:
    job = execute(circ, Aer.get_backend('qasm_simulator'), shots=10000)
    results.append(job.result())
    print(job.result().get_counts())
    print(trace_out_ancillae(job.result().get_counts()))
