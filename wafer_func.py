from sklearn.datasets import make_blobs
from wafer import *


# 1. Random
def C1():
  coord = noise()

  return coord


# 2. Wafer edge ring defect
def C2():
    random = noise()

    n_ring = np.random.uniform(100, 300)

    r_0 = 10
    delta = np.random.uniform(1, 3)

    x_li, y_li = [], []
    for _ in range(int(n_ring)):
        a_ring = np.random.uniform(0, 2 * np.pi)
        r_ring = np.random.uniform(r_0 - delta, r_0 )

        x, y = cartesian(a_ring, r_ring)

        x_li.append(x)
        y_li.append(y)

    coord = np.array(list(zip(x_li, y_li)))
    coord = filtered(np.concatenate((random, coord), axis=0))

    return coord


# 3. Wafer right side edge defect
def C3():
    random = noise()

    n_ring = np.random.uniform(150, 300)

    r_0 = 10
    delta = np.random.uniform(1, 4)

    x_li, y_li = [], []
    t1 = np.random.uniform(-np.pi / 2, -np.pi/10)
    t2 = np.random.uniform(np.pi/10, np.pi / 2)
    for _ in range(int(n_ring)):
        a_ring = np.random.uniform(t1, t2)
        r_ring = np.random.uniform(r_0 - delta, r_0 + delta)

        x, y = cartesian(a_ring, r_ring)

        x_li.append(x)
        y_li.append(y)

    coord = np.array(list(zip(x_li, y_li)))
    coord = filtered(np.concatenate((random, coord), axis=0))

    return coord


# 4. Wafer left side edge defect
def C4():
    random = noise()

    n_ring = np.random.uniform(150, 300)

    r_0 = 10
    delta = np.random.uniform(1, 4)

    x_li, y_li = [], []
    t1 = np.random.uniform(np.pi/2 + np.pi / 2, np.pi/2 + -np.pi/10)
    t2 = np.random.uniform(np.pi + np.pi / 10, np.pi +  np.pi / 2)
    for _ in range(int(n_ring)):
        a_ring = np.random.uniform(t1, t2)
        r_ring = np.random.uniform(r_0 - delta, r_0 + delta)

        x, y = cartesian(a_ring, r_ring)

        x_li.append(x)
        y_li.append(y)

    coord = np.array(list(zip(x_li, y_li)))
    coord = filtered(np.concatenate((random, coord), axis=0))

    return coord


# 5. Line scratch
def C5(show=False):
    random = noise()

    while True:
        a, b = np.random.uniform(-10, 10), np.random.uniform(-10, 10)
        if abs(a - b) > 5:
            a, b = sorted([a, b])
            break

    while True:
        k = np.random.uniform(-1/15, 1/15)
        if k != 0:
            break

    n_scratch = np.random.uniform(50, 100)

    x_i = np.linspace(a, b, int(n_scratch))
    y_i = k * x_i

    a = np.random.uniform(0, 2 * np.pi)
    x = np.cos(a)*x_i - np.sin(a)*y_i
    y = np.sin(a)*x_i + np.cos(a)*y_i

    x, y = np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)

    coord = np.concatenate((x, y), axis=1)
    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 6. Curved scratch defect
def C6():
    random = noise()

    while True:
        a, b = np.random.uniform(-10, 10), np.random.uniform(-10, 10)
        if abs(a - b) > 5:
            a, b = sorted([a, b])
            break

    while True:
        k = np.random.uniform(-1/15, 1/15)
        if k != 0:
            break

    n_scratch = np.random.uniform(50, 100)

    x_i = np.linspace(a, b, int(n_scratch))
    y_i = k * x_i**2

    a = np.random.uniform(0, 2 * np.pi)
    x = np.cos(a)*x_i - np.sin(a)*y_i
    y = np.sin(a)*x_i + np.cos(a)*y_i

    x, y = np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)

    coord = np.concatenate((x, y), axis=1)
    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 7. Non-random cluster defect at 1st quadrant
def C7():
    random = noise()

    n_cluster = int(np.random.uniform(150, 300))
    cluster = 1
    cluster_std = np.random.uniform(0.1, 2)

    coord, _ = make_blobs(n_samples=n_cluster, n_features=2, centers=cluster, cluster_std=cluster_std, center_box=(4, 5))

    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 8. Non-random cluster defect at 2nd quadrant
def C8():
    random = noise()

    c7 = C7()
    c7[:, 0] = -c7[:, 0]

    coord = np.concatenate((c7, random), axis=0)

    coord = filtered(coord)

    return coord


# 9. Non-random cluster defect at 3rd quadrant
def C9():
    random = noise()

    n_cluster = int(np.random.uniform(150, 300))
    cluster = 1
    cluster_std = np.random.uniform(0.1, 2)

    coord, _ = make_blobs(n_samples=n_cluster, n_features=2, centers=cluster, cluster_std=cluster_std, center_box=(-5, -4))

    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 10. Non-random cluster defect at 4th quadrant

def C10():
    random = noise()

    c7 = C7()
    c7[:, 1] = -c7[:, 1]

    coord = np.concatenate((c7, random), axis=0)

    coord = filtered(coord)

    return coord


# 11. Non-random cluster defect at center
def C11():
    random = noise()

    n_cluster = int(np.random.uniform(150, 300))
    cluster = 1
    cluster_std = np.random.uniform(0.1, 2)

    coord, _ = make_blobs(n_samples=n_cluster, n_features=2, centers=cluster, cluster_std=cluster_std, center_box=(0, 0))

    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 12. Non-random cluster defect at top
def C12():
    random = noise()

    n_cluster = int(np.random.uniform(150, 300))
    cluster_std = np.random.uniform(0.1, 2)

    y = np.random.uniform(4, 5)
    coord, _ = make_blobs(n_samples=n_cluster, n_features=2, centers=[(0, y)], cluster_std=cluster_std)

    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 13. Non-random cluster defect at down
def C13():
    random = noise()

    c12 = C12()
    c12[:, 1] = -c12[:, 1]

    coord = np.concatenate((c12, random), axis=0)

    coord = filtered(coord)

    return coord


# 14. Non-random cluster defect at right
def C14():
    random = noise()

    n_cluster = int(np.random.uniform(150, 300))
    cluster = 1
    cluster_std = np.random.uniform(0.1, 2)

    x = np.random.uniform(4, 5)
    coord, _ = make_blobs(n_samples=n_cluster, n_features=2, centers=[(x, 0)], cluster_std=cluster_std)

    coord = np.concatenate((coord, random), axis=0)

    coord = filtered(coord)

    return coord


# 15. Non-random cluster defect at left
def C15():
    random = noise()

    c14 = C14()
    c14[:, 0] = -c14[:, 0]

    coord = np.concatenate((c14, random), axis=0)

    coord = filtered(coord)

    return coord


# 16. Gross defect at entire wafer
def C16():
    random = noise()
    n_dense = noise(150, 300)

    coord = np.concatenate((random, n_dense), axis=0)

    return coord


# 17. Gross defect at left half of wafer
def C17():
    random = noise()
    n_dense = noise(150, 300)

    n_dense = n_dense[n_dense[:, 0] <= 0]

    coord = np.concatenate((random, n_dense), axis=0)

    return coord


# 18. Gross defect at right half of wafer
def C18():
    random = noise()

    c17 = C17()
    c17[:, 0] = -c17[:, 0]

    coord = np.concatenate((c17, random), axis=0)

    return coord


# 19. Line scratch with non-random cluster defect at 1st quadrant
def C19():
    c5 = C5()
    c7 = C7()

    coord = np.concatenate((c5, c7), axis=0)

    return coord


# 20. Line scratch with non-random cluster defect at 2nd quadrant
def C20():
    c5 = C5()
    c8 = C8()

    coord = np.concatenate((c5, c8), axis=0)

    return coord


# 21. Line scratch with non-random cluster defect at 3rd quadrant
def C21(show=False):
    c5 = C5()
    c9 = C9()

    coord = np.concatenate((c5, c9), axis=0)

    return coord


# 22. Line scratch with non-random cluster defect at 4th quadrant
def C22(show=False):
    c5 = C5()
    c10 = C10()

    coord = np.concatenate((c5, c10), axis=0)

    return coord
