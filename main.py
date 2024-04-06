import os
import argparse
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import create_dataset
from model import MyModel


def get_args_parser():
    parser = argparse.ArgumentParser('Locational Persistence Images', add_help=False)
    parser.add_argument('--pi_path', default='PI_data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--save_model', default='results.pt', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--input_size', default=300, type=int)
    parser.add_argument('--save_acc_name', default='LPI', type=str)

    return parser


def main(args):
    # 데이터 폴더 경로
    data_dir = args.pi_path


    model = MyModel(input_size=args.input_size)
    device = torch.device(args.device)


    if args.test:
        test_dataset = create_dataset(os.path.join(data_dir, 'valid'), "test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        model.load_state_dict(torch.load(args.save_model))
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


    # 데이터셋 생성
    train_dataset = create_dataset(os.path.join(data_dir, 'train'), "train")
    val_dataset = create_dataset(os.path.join(data_dir, 'valid'), "valid")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)


    # 손실 함수와 최적화 함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-7)


    model.to(device)

    val_accuracies = []
    train_accuracies = []

    print("Start Training & Evaluation")
    start_time = time.time()
    # 훈련 루프
    num_epochs = args.epoch
    max_acc = -1.0
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total


        # eval 루프
        model.eval()
        correct = 0
        total = 0
        total_val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')

        train_accuracies.append(train_acc)
        val_accuracies.append(val_accuracy)


        if max_acc < val_accuracy:
            max_acc = val_accuracy
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
    parser = argparse.ArgumentParser('Locational Persistence Images training and inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)