import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import time
import datetime
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('Training with Pre-trained ResNet50', add_help=False)
    parser.add_argument('--img_path', default='img_data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_model', default='results_resnet50.pt', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ex_test', action='store_true', help='ture when small or imbalance experiments')
    parser.add_argument('--save_acc_name', default='CNN',type=str)

    return parser


def main(args):
    # 이미지 전처리를 위한 변환 함수를 정의
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 데이터셋 폴더 경로
    data_dir = args.img_path
    batch_size = args.batch_size


    model = models.resnet50(pretrained=True)

    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 22)
    )

    # Set requires_grad to True for the new layers
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


    device = torch.device(args.device)


    if args.test:
        test_dir = os.path.join(data_dir, "test")
        test_dataset = ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)


        checkpoint = torch.load(args.save_model)
        
        # model.state_dict의 key와 일치하지 않는 문제 발생
        if not args.ex_test:
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]

        model.load_state_dict(checkpoint)
        model.to(device)

        # 모델 평가
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}')
        return


    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")


    # Dataset
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)


    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-7,
    )

    model.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model)

    val_accuracies = []
    train_accuracies = []

    max_acc = -1
    num_epochs = args.epoch

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Training [{epoch+1}/{num_epochs}]', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total


        # 검증 데이터 평가
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_acc:.2f}% | Validation Accuracy: {val_acc:.2f}%')

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if max_acc < val_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), args.save_model)

        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training & Evaluation time {}'.format(total_time_str))


    with open(f'{args.save_acc_name}_train_acc.txt', 'w') as f:
        for acc in train_accuracies:
            f.write(f'{acc}\n')

    with open(f'{args.save_acc_name}_val_acc.txt', 'w') as f:
        for acc in val_accuracies:
            f.write(f'{acc}\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser('ResNet50 transfer learning and inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)