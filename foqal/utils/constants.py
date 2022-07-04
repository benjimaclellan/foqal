import numpy as np
import qutip as qt

"""
Common quantum states
"""

states = {
    "H": qt.Qobj(np.array([1.0, 0.0])),
    "V": qt.Qobj(np.array([0.0, 1.0])),
    "D": qt.Qobj(np.array([1.0, 1.0]) / np.sqrt(2)),
    "A": qt.Qobj(np.array([1.0, -1.0]) / np.sqrt(2)),
    "R": qt.Qobj(np.array([1.0, 1.0j]) / np.sqrt(2)),
    "L": qt.Qobj(np.array([1.0, -1.0j]) / np.sqrt(2)),
    "phi+": qt.Qobj(np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2), dims=[[2, 2], [1, 1]]),
    "phi-": qt.Qobj(
        np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2), dims=[[2, 2], [1, 1]]
    ),
    "psi+": qt.Qobj(np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2), dims=[[2, 2], [1, 1]]),
    "psi-": qt.Qobj(
        np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2), dims=[[2, 2], [1, 1]]
    ),
}


""" Common quantum state models """


def depolarized(p, state=states['phi+']):
    identity = qt.identity(dims=[2, 2])
    rho_prime = (1-p) * qt.ket2dm(state) + p * identity / 4
    return rho_prime


# def dephased(p, state=states['phi+']):
#     z = qt.tensor(qt.identity(2), qt.sigmaz())
#     K0 = qt.Qobj(np.array([[1.0, 0.0],
#                            [0.0, np.sqrt(1 - p)]]))
#     K1 = qt.Qobj(np.array([[0.0, np.sqrt(p)],
#                            [0.0, 0.0]]))
#     rho = qt.ket2dm(state).unit()
#     rho_prime = qt.tensor(identity, K0) * rho * qt.tensor(identity, K0.dag()) \
#                 + qt.tensor(identity, K1) * rho * qt.tensor(identity, K1.dag())
#     return rho_prime


# def amplitude_damping(p, state=states['phi+']):
#     identity = qt.identity(dims=[2, ])
#     K0 = qt.Qobj(np.array([[1.0, 0.0],
#                            [0.0, np.sqrt(1-p)]]))
#     K1 = qt.Qobj(np.array([[0.0, np.sqrt(p)],
#                            [0.0, 0.0]]))
#     rho = qt.ket2dm(state).unit()
#     rho_prime = qt.tensor(identity, K0) * rho * qt.tensor(identity, K0.dag()) \
#                 + qt.tensor(identity, K1) * rho * qt.tensor(identity, K1.dag())
#     return rho_prime


channels = {
    "depolarized": depolarized,
}
