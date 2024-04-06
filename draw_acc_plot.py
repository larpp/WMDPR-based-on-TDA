import matplotlib.pyplot as plt
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Plot Training/Validation accuracy', add_help=False)
    parser.add_argument('--LPI_train', default='LPI_train_acc.txt', type=str)
    parser.add_argument('--LPI_val', default='LPI_val_acc.txt', type=str)
    parser.add_argument('--CNN_train', default='CNN_train_acc.txt', type=str)
    parser.add_argument('--CNN_val', default='CNN_val_acc.txt', type=str)
    parser.add_argument('--TDA_train', default='TDA_train_acc.txt', type=str)
    parser.add_argument('--TDA_val', default='TDA_val_acc.txt', type=str)

    return parser


# 파일에서 데이터 읽기
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data


def main(args):
    lpi_train_acc = read_data(args.LPI_train)
    lpi_val_acc = read_data(args.LPI_val)

    cnn_train_acc = read_data(args.CNN_train)
    cnn_val_acc = read_data(args.CNN_val)

    tda_train_acc = read_data(args.TDA_train)
    tda_val_acc = read_data(args.TDA_val)


    # 손실과 정확도 그래프 그리기
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(cnn_train_acc, label='Traininig')
    plt.plot(cnn_val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim((0, 102))
    plt.title('CNN')

    plt.subplot(1, 3, 2)
    plt.plot(tda_train_acc, label='Training')
    plt.plot(tda_val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim((0, 102))
    plt.title('TDA')

    plt.subplot(1, 3, 3)
    plt.plot(lpi_train_acc, label='Training')
    plt.plot(lpi_val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim((0, 102))
    plt.title('LPI')

    plt.savefig('results.png')
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser('Plot accuracy and save script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)