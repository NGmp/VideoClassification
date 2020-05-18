import torch
from torch import nn


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=10):
            super(VideoClassifier, self).__init__()

            self.conv1 = nn.Conv3d(in_channels=3,out_channels=96,kernel_size=5, stride=1)
            self.norm = nn.BatchNorm3d(96)
            self.pool = nn.AvgPool3d(kernel_size=2)
            self.conv2 = nn.Conv3d(in_channels=96, out_channels=256, kernel_size= 3, stride=1)
            self.norm2 = nn.BatchNorm3d(256)
            self.pool2 = nn.AvgPool3d(kernel_size=2)
            self.conv3 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size= 3 , stride = 1)
            self.pool3 = nn.AvgPool3d(kernel_size=2)
            self.fc1 = nn.Linear(in_features=270848, out_features= 4096)
            self.fc2 = nn.Linear(in_features= 4096, out_features=num_classes)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.pool(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.fc2(x)
        return x
