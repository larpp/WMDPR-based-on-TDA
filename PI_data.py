import os
import numpy as np
import argparse

from img2pi import PI
from wafer_func import *
from imbalance_num import imbalance


def get_args_parser():
    parser = argparse.ArgumentParser('Number of Datasets', add_help=False)
    parser.add_argument('--path', default='PI_data', type=str)
    parser.add_argument('--train_num', default=300, type=int)
    parser.add_argument('--val_num', default=100, type=int)
    parser.add_argument('--test_num', default=100, type=int)
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lpi', action='store_false')

    return parser


# Class 이름으로 된 디렉터리 생성
def make_class(path, mode):
    for i in range(1, 23):
        os.makedirs(f"{path}/{mode}/C{i}")


# Image (defect 좌표)를 PI로 변환 후 저장
def make_dataset(args):
    path = args.path

    if args.imbalance:
        train = imbalance(args.seed)
        print(train)

        def save_train_dataset(train_li=train, mode="train"):
            make_class(path, mode)

            classes = os.listdir(f'{path}/{mode}')
            classes = sorted(classes, key=lambda x: int(x[1:]))

            for i, folder_name in enumerate(classes):
                if i == 0: num = 0
                else: num += train_li[i-1]
                for k in range(train_li[i]):
                    function_name = f'C{i + 1}'
                    func = globals()[function_name]
                    coord = func()
                    vec = PI(coord, args.lpi)
                    np.savetxt(f"{path}/{mode}/{folder_name}/{k+1 + num:05d}.csv", vec)

            return sum(train_li)

    else:
        train = args.train_num

        def save_train_dataset(n=train, mode="train"):
            make_class(path, mode)

            classes = os.listdir(f'{path}/{mode}')
            classes = sorted(classes, key=lambda x: int(x[1:]))

            for i, folder_name in enumerate(classes):
                for k in range(1, n+1):
                    function_name = f'C{i + 1}'
                    func = globals()[function_name]
                    coord = func()
                    vec = PI(coord, args.lpi)
                    np.savetxt(f"{path}/{mode}/{folder_name}/{k + n * i:05d}.csv", vec)

            return n * 22

    train_num = save_train_dataset()

    val = args.val_num
    test = args.test_num

    def save_val_dataset(num=train_num, n=val, mode="valid"):
        make_class(path, mode)

        train_len = num

        classes = os.listdir(f'{path}/{mode}')
        classes = sorted(classes, key=lambda x: int(x[1:]))

        for i, folder_name in enumerate(classes):
            for k in range(1, n+1):
                function_name = f'C{i + 1}'
                func = globals()[function_name]
                coord = func()
                vec = PI(coord, args.lpi)
                np.savetxt(f"{path}/{mode}/{folder_name}/{train_len + k + n * i:05d}.csv", vec)

        return num + (22 * n)


    train_val_num = save_val_dataset()


    def save_test_dataset(num=train_val_num, n=test, mode="test"):
        make_class(path, mode)

        num = train_val_num

        classes = os.listdir(f'{path}/{mode}')
        classes = sorted(classes, key=lambda x: int(x[1:]))

        for i, folder_name in enumerate(classes):
            for k in range(1, n+1):
                function_name = f'C{i + 1}'
                func = globals()[function_name]
                coord = func()
                vec = PI(coord, args.lpi)
                np.savetxt(f"{path}/{mode}/{folder_name}/{num + k + n * i:05d}.csv", vec)

    save_test_dataset()

    print("PI Data Generation Complete!!")


if __name__=='__main__':
    parser = argparse.ArgumentParser('Set Datasets number script', parents=[get_args_parser()])
    args = parser.parse_args()
    make_dataset(args)