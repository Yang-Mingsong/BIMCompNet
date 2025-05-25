import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, num_classes):
        super(VoxNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            # Input: (1, 256, 256, 256)
            ('conv1', nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2)),
            ('bn1', nn.BatchNorm3d(32)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool3d(2)),

            ('conv2', nn.Conv3d(32, 64, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm3d(64)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool3d(2)),

            ('conv3', nn.Conv3d(64, 128, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm3d(128)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool3d(2)),

            ('conv4', nn.Conv3d(128, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm3d(256)),
            ('relu4', nn.ReLU()),
            ('pool4', nn.MaxPool3d(2)),
        ]))

        # 256 * 8 * 8 * 8 = 131072
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256 * 8 * 8 * 8, 512)),
            ('relu5', nn.ReLU()),
            ('drop1', nn.Dropout(0.4)),
            ('fc2', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
