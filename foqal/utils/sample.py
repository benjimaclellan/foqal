import numpy as np
import qutip as qt


def bloch_vector_to_ket(vector):
    (x, y, z) = vector
    theta = np.arccos(z) / 2
    phi = np.arctan2(y, x)
    ket = qt.Qobj(
        np.array([[np.cos(theta)], [np.exp(1j * phi) * np.sin(theta)]]),
        # dims=((2,), (1,),)
    )
    return ket


def bloch_vectors_to_kets(vectors):
    kets = [bloch_vector_to_ket(vector) for vector in vectors]
    return kets


def distribute_points_on_sphere(num_samples: int = 1000, method: str = "spiral"):
    """

    Acknowledgements to https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Parameters
    ----------
    num_samples: int Number of samples to distribute on the surface of the sphere
    method: can be either 'spiral' or 'haar'

    Returns
    -------

    """

    if method == "spiral":
        raise ValueError(
            "Spiral sampling method not implemented currently (needs more testing)."
        )
        N = num_samples
        C = 3.6

        # values for theta and phi are determined according to Rakhmanov, Saff, and Zhou
        phi = 0
        vectors = []
        for k in range(1, N + 1):
            h = -1 + (2 * (k - 1)) / (N - 1)
            theta = np.arccos(h)

            if k == 1:  # we require that phi_0 = phi_{N} = 0
                x = 0
                y = 0
                z = -1
            elif k == N:
                x = 0
                y = 0
                z = 1
            else:
                phi += (C / np.sqrt(N)) * (1 / np.sqrt((1 - h**2)))
                x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)

            vectors.append((x, y, z))

    elif method == "fibonnaci":
        vectors = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        for i in range(num_samples):
            z = 1 - (i / float(num_samples)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - z * z)  # radius at z

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            y = np.sin(theta) * radius

            vectors.append((x, y, z))

    elif method == "haar":
        vectors = []
        for i in range(num_samples):
            vector = np.random.uniform(-1, 1, 3)
            vector = vector / np.sqrt(np.sum(np.power(vector, 2)))
            vectors.append(tuple(vector))

    else:
        raise NotImplementedError("Options are 'spiral', 'fibonnaci', or 'haar'.")

    return vectors
