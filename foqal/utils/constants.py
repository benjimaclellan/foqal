import numpy as np
import qutip as qt

"""
Common quantum states as qt.QObjs in an easily-accessible dictionary.
Example usage,
    from qoqi.utils.constants import states
    print(states['H'])
    print(states['psi+'])
"""

# common quantum states of light as Qobjs
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
    "HH": qt.Qobj(np.array([1.0, 0.0, 0.0, 0.0]), dims=[[2, 2], [1, 1]]),
    "HV": qt.Qobj(np.array([0.0, 1.0, 0.0, 0.0]), dims=[[2, 2], [1, 1]]),
    "VH": qt.Qobj(np.array([0.0, 0.0, 1.0, 0.0]), dims=[[2, 2], [1, 1]]),
    "VV": qt.Qobj(np.array([0.0, 0.0, 0.0, 1.0]), dims=[[2, 2], [1, 1]]),
}
