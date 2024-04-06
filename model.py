import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size=300, hidden_size=1024, class_num=22):
        super(MyModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, class_num)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x