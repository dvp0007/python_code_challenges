import numpy as np


def helper_1(v, C):
    C1 = np.linalg.inv(
        np.matrix([[C[0], C[5], C[4]], [C[5], C[1], C[3]], [C[4], C[3], C[2]]]))
    return [C1[0, 0], C1[1, 1], C1[2, 2], C1[1, 2], C1[0, 2], C1[0, 1]], v


def helper_2(v, C):
    I1 = C[0] + C[1] + C[2]
    I2 = (-C[0]**2 - C[1]**2 - C[2]**2 - 2*C[3]**2 - 2*C[4]
          ** 2 - 2*C[5]**2 + (C[0] + C[1] + C[2]) ** 2)/2
    I3 = C[0]*C[1]*C[2] - C[0]*C[3]**2 - C[1] * \
        C[4]**2 - C[2]*C[5]**2 + 2*C[3]*C[4]*C[5]
    invariants_list = [I1, I2, I3]
    return invariants_list, v


def test_many_vectors(vs, C):
    accumulator = np.zeros((6))
    for v in vs:
        l, v = helper_2(v, C)
        trace, _, det = l
        C_inv, v = helper_1(v, C)
        I = np.array([1, 1, 1, 0, 0, 0])
        D1, D2, D3 = I, trace*I-C, det*np.array(C_inv)
        D_mat = np.array([D1, D2, D3]).transpose()
        accumulator += np.dot(D_mat, v)
    return accumulator / len(vs)


if __name__ == '__main__':
    shear = np.random.uniform(0.01, 0.2)
    C = np.array([1 + shear**2, 1, 1, 0, 0, shear])

    vs = [np.random.rand(3) for i in range(100000)]

    print(test_many_vectors(vs, C))
