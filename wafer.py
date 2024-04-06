import matplotlib.pyplot as plt
import numpy as np


# Wafer 틀
def wafer():
    plt.figure(figsize=(5,5))
    plt.axis('off')

    radius = 10
    margin = 1

    x = np.linspace(-radius - margin, radius + margin, 100)
    y = np.linspace(-radius - margin, radius + margin, 100)

    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - radius ** 2

    plt.contour(X, Y, F, [0])


# Wafer 원틀 함수
def f(x, y):
      return x ** 2 + y ** 2 - 10 ** 2


# Wafer 밖에 있는 점들을 걸러내기 위한 함수
def filtered(coord):
    idx = np.where(f(coord[:,0], coord[:,1]) < 0)
    li = []
    for i in idx:
        li.append(coord[idx])
    return np.array(*li)


# 극좌표를 직교좌표로 바꾸는 함수
def cartesian(a, r):
    x = r * np.cos(a)
    y = r * np.sin(a)
    return x, y


# Wafer의 random defect points를 생성해주는 함수
def noise(start=10, end=60):
    n_random = np.random.uniform(start, end)

    x_li, y_li = [], []
    for _ in range(int(n_random)):
        a_random = np.random.uniform(0, 2 * np.pi)
        r_random = np.random.uniform(0, 10)

        x, y = cartesian(a_random, r_random)

        x_li.append(x)
        y_li.append(y)

    return np.array(list(zip(x_li, y_li)))


# wafer 그림을 보여주는 함수
def show_wafer(coord):
    plt.scatter(coord[:, 0], coord[:, 1], marker='s', s=1)
    plt.show()


# 그림 저장 함수
def save_wafer_plot(coord, filename):
    plt.figure()
    wafer()
    plt.scatter(coord[:, 0], coord[:, 1], marker='s', s=1)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()