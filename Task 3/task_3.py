import numpy as np
import tqdm


def mark_pores_slow(x, pore_centers, pore_radii):
    I, J, K = x.shape
    for i in tqdm.trange(I):
        for j in range(J):
            for k in range(K):
                position = np.array([i, j, k])
                for pore_center, pore_radius in zip(pore_centers, pore_radii):
                    delta = pore_center - position
                    distance = np.sqrt(np.dot(delta, delta))
                    if distance <= pore_radius:
                        x[i, j, k] = 1
    return x


def mark_pores_fast(x, pore_centers, pore_radii):
    """
    This is for you to implement. It should yield the same result as the above function
    mark_pores_slow, but it should run faster.
    Hint: If you do it correctly, you should be able to change the below variable 'res'
    from 100 to 1000 without a significant slow-down of mark_pores_fast.

    Args:
        x: Numpy array of spatial domain in 3D. Initialized with 0.
        pore_centers: List of numpy arrays representing the centers of spherical pores
        pore_radii: List of pore radii

    Returns:
        x: Each field in x is 1 if in pore, 0 else.
    """
    raise NotImplementedError('This is your job')
    return x


if __name__ == '__main__':

    # initialize domain
    res = 100
    x = np.zeros((res, res, res))

    # sample some random pores
    n_pores = 10
    pore_radii = [np.random.uniform(3, 10) for n_pore in range(n_pores)]
    pore_centers = [np.random.rand(3) * x.shape for n_pore in range(n_pores)]

    # mark pores in domain
    pore_marker_slow = mark_pores_slow(x, pore_centers, pore_radii)
    pore_marker_fast = mark_pores_fast(x, pore_centers, pore_radii)
