import matplotlib.pyplot as plt
from wafer_func import *
import argparse

from ripser import Rips
from persim import PersistenceImager

import collections
collections.Iterable = collections.abc.Iterable

import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Latex 사용하지 않도록 강제 설정


def get_args_parser():
    parser = argparse.ArgumentParser('Persistence Images', add_help=False)
    parser.add_argument('--label', default='1', type=int)

    return parser


def H0(coord, title):
    rips = Rips()
    # dgms에는 H0의 PD에서의 좌표가 들어가 있다.
    dgms = rips.fit_transform(coord)
    H0_dgm = dgms[0][:-1,:]  # 무한대 제외

    # H0
    pimgr0 = PersistenceImager(pixel_size=1, birth_range=(0,10), pers_range=(0.0, 10.0))
    pimgr0.kernel_params = {'sigma': .1}
    # pimgr.fit(H1_dgm)

    # skew=False하면 좌표 변환이 안됨
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    pimgr0.plot_diagram(H0_dgm, skew=True, ax=axs[0])
    axs[0].set_title('Diagram', fontsize=16)

    pimgr0.plot_image(pimgr0.transform(H0_dgm), ax=axs[1])
    axs[1].set_title('Pixel Size: 1', fontsize=16)

    plt.savefig(f'{title}_H0.png')


def H1(coord, title):
    rips = Rips()
    # dgms에는 H1의 PD에서의 좌표가 들어가 있다.
    dgms = rips.fit_transform(coord)
    H1_dgm = dgms[1]

    # H1
    pimgr1 = PersistenceImager(pixel_size=1, birth_range=(0,10), pers_range=(0.0, 10))
    # 분산 조절하는 부분
    pimgr1.kernel_params = {'sigma': 0.1}

    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    pimgr1.plot_diagram(H1_dgm, skew=True, ax=axs[0])
    axs[0].set_title('Diagram', fontsize=16)

    pimgr1.plot_image(pimgr1.transform(H1_dgm), ax=axs[1])
    axs[1].set_title('Pixel Size: 1', fontsize=16)

    plt.savefig(f'{title}_H1.png')


def LPI(coord, title):
    # Locational_PI
    # 원본 이미지의 좌표 점을 0보다 크게 만들어주기 위해 좌표를 이동시킨다. (반지름이 10이므로 10씩 이동)
    # 또한 거기에 맞게 birth_range (x축으로 간주)와 pers_range (y축으로 간주)를 바꾼다.

    pimgr = PersistenceImager(pixel_size=2, birth_range=(0, 20), pers_range=(0, 20))
    pimgr.kernel_params = {'sigma': 0.1}
    pimgr.weight_params = {'n' : 0}

    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].scatter(coord[:, 0]+10, coord[:, 1]+10, color='blue')
    axs[0].set_title('Original shape', fontsize=16)
    # 그래프의 x, y축의 범위를 맞춰줌
    axs[0].set_xlim(0, 20)
    axs[0].set_ylim(0, 20)

    # 원본 이미지와 맞춰주기 위해서 x, y에 각각 10씩 더해줌
    # skew=True로 해버리면 좌표축이 변형되므로 False로 해준다.
    pimgr.plot_image(pimgr.transform(coord+10, skew=False), ax=axs[1])
    axs[1].set_title('Pixel Size: 1', fontsize=16)

    # 가로 세로 비율을 조정
    axs[0].set_aspect('equal')

    plt.savefig(f'{title}_LPI.png')


def main(args):
    function_name = f'C{args.label}'
    func = globals()[function_name]
    coord = func()

    H0(coord, function_name)
    H1(coord, function_name)
    LPI(coord, function_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser('Practice Persistence Images script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)