import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('PCA and tSNE results', add_help=False)
    parser.add_argument('--path', default='PI_data', type=str)
    parser.add_argument('--type', default='LPI',type=str)

    return parser


def main(args):
    data_path = args.path

    data = []
    class_labels = []


    for class_label in range(1, 23):
        class_name = f'C{class_label}'
        path = f'{data_path}/train/{class_name}'
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            value= np.loadtxt(file_path)
            data.append(value)
            class_labels.append(class_name)


    X = np.vstack(data)
    class_labels = np.array(class_labels)


    # Data Normalization
    scaler = StandardScaler()
    scaler.fit(X)
    scaled = scaler.transform(X)


    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)


    plt.figure(figsize=(6, 6))
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple', 'pink', 'green', 'brown', 'gray', 'blue',
            'lime', 'darkviolet', 'olive', 'sienna', 'lightpink', 'gold', 'cyan', 'yellow', 'orchid', 'silver', 'teal', 'magenta']


    labels = np.unique(class_labels)
    labels = sorted(labels, key=lambda x: int(x[1:]))
    for color, label in zip(colors, labels):
        plt.scatter(X_r[class_labels == label, 0], X_r[class_labels == label, 1], color=color, alpha=.8, lw=2, label=label)

    plt.legend(loc=(0.97, 0.0), shadow=False, scatterpoints=1)
    plt.title(f'PCA of {args.type}')
    plt.savefig(f'{args.type}_PCA')
    plt.close()


    plt.figure(figsize=(6, 6))

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
    X_2d = tsne.fit_transform(scaled)

    for color, label in zip(colors, labels):
        plt.scatter(X_2d[class_labels == label, 0], X_2d[class_labels == label, 1], color=color, alpha=.8, lw=2, label=label)

    plt.legend(loc=(0.97, 0.0), shadow=False, scatterpoints=1)
    plt.title(f'tSNE of {args.type}')
    plt.savefig(f"{args.type}_tSNE")
    plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser('Plot PCA and tSNE of LPI and CNN script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)