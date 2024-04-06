import os
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import csv
from img_data import make_class
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Save CNN features', add_help=False)
    parser.add_argument('--path', default='img_data', type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--save_dir', default='CNN_features', type=str)

    return parser


def main(args):
    transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    # 데이터셋 폴더 경로
    data_dir = args.path


    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])

    device = torch.device(args.device)

    train_dir = os.path.join(data_dir, "train")
    train_dataset = ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers)

    model.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model)


    make_class(args.save_dir, "train")
    idx = 0
    for images, labels in train_loader:
        idx += 1
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.squeeze().cpu().tolist()
        labels.cpu().tolist()

        # 리스트를 CSV 파일로 저장
        with open(f'{args.save_dir}/train/C{labels[0]+1}/{idx:05d}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for item in outputs:
                csvwriter.writerow([item])


if __name__=='__main__':
    parser = argparse.ArgumentParser('Save features of ResNet50 script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)