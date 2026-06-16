import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights


class SVCNN(nn.Module):

    def __init__(self, num_classes, pretraining=True, cnn_name=""):
        super(SVCNN, self).__init__()

        self.num_classes = num_classes
        self.pretraining = pretraining
        self.cnn_name = cnn_name

        if self.cnn_name == 'alexnet':
            self.net = models.alexnet(pretrained=self.pretraining)
            self.net.classifier._modules['6'] = nn.Linear(4096, num_classes)

        elif self.cnn_name.startswith("resnet"):
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, num_classes)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34( pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, num_classes)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, num_classes)
            elif self.cnn_name == 'resnet101':
                self.net = models.resnet101(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, num_classes)

        elif self.cnn_name.startswith("mobilenet"):
            if self.cnn_name == 'mobilenet_v3_large':
                self.net = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, reduced_tail=False, dilated=False)
                self.net.classifier._modules['3'] = nn.Linear(1280, num_classes)
            elif self.cnn_name == 'mobilenet_v3_small':
                self.net = models.mobilenet_v3_small(pretrained=self.pretraining)
                self.net.classifier._modules['3'] = nn.Linear(1024, num_classes)

        elif self.cnn_name.startswith("vgg"):
            if self.cnn_name == 'vgg11':
                self.net = models.vgg11(pretrained=self.pretraining)
            elif self.cnn_name == 'vgg13':
                self.net = models.vgg13(pretrained=self.pretraining)
            elif self.cnn_name == 'vgg16':
                self.net = models.vgg16(pretrained=self.pretraining)
            elif self.cnn_name == 'vgg19':
                self.net = models.vgg19(pretrained=self.pretraining)
            self.net.classifier._modules['6'] = nn.Linear(4096, num_classes)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)


class MVCNN(nn.Module):

    def __init__(self, svcnn_model, num_classes, num_views):
        super(MVCNN, self).__init__()

        self.num_classes = num_classes
        self.num_views = num_views

        if svcnn_model.cnn_name.startswith(("alexnet", "mobilenet", "vgg")):
            self.net_1 = nn.Sequential(*list(svcnn_model.net.children())[:-1])
            self.net_2 = svcnn_model.net.classifier
        if svcnn_model.cnn_name.startswith('resnet'):
            self.net_1 = nn.Sequential(*list(svcnn_model.net.children())[:-1])
            self.net_2 = svcnn_model.net.fc

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        y = torch.max(y, 1)[0]
        y = y.view(y.shape[0], -1)
        y = self.net_2(y)
        return y
