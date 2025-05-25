import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.GroupNorm(32, 64)
        self.bn2 = nn.GroupNorm(32, 128)
        self.bn3 = nn.GroupNorm(32, 1024)
        self.bn4 = nn.GroupNorm(32, 512)
        self.bn5 = nn.GroupNorm(32, 256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array(
            [1, 0, 0, 0, 1, 0, 0, 0, 1]
        ).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNet(nn.Module):
    # 3 个通道为x, y, z
    def __init__(self, num_classes, channel):
        super(PointNet, self).__init__()
        # 变换矩阵
        self.stn = STN3d(3)
        # 升维度到1024维
        self.encoder = nn.Sequential(
            nn.Conv1d(channel, 64, 1),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.GroupNorm(32, 128),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.GroupNorm(32, 1024)
        )

        # 全局最大池化层，用于聚合点的特征
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # 分类层（如果需要）
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.encoder(x)
        x = self.maxpool(x).squeeze(-1)
        x = self.classifier(x)
        return x
