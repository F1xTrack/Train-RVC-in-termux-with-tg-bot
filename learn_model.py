import torch
import torch.nn as nn

class SimpleRVCModel(nn.Module):
    def __init__(self):
        super(SimpleRVCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)  # Выход на 10 классов

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Уплощаем тензор
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
